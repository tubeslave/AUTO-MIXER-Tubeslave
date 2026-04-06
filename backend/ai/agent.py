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
        # Lazy imports to avoid circular deps
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
        self.kb_first = kb_first
        self.state = AgentState(mode=mode)
        self.cycle_interval = cycle_interval
        self._confidence_threshold = 0.6
        self._max_actions_per_cycle = 10
        self._action_history: List[AgentAction] = []
        self._llm_query_interval = 10.0
        self._last_llm_query = 0.0
        # cooldown: track last time each (channel, action_type) was applied
        self._last_applied: Dict[tuple, float] = {}
        # cooldowns per action type (seconds before re-firing same action on same ch)
        self._cooldowns: Dict[str, float] = {
            "apply_hpf":            300.0,  # structural, once per session
            "apply_eq":              45.0,  # re-analyze every 45s (signal may change)
            "adjust_compressor":     60.0,  # re-check compression every minute
            "adjust_compression":    60.0,  # threshold fine-tune
            "apply_phase_correction": 30.0, # phase — after bleed analysis
            "reduce_gain":            8.0,  # level guard, reactive
            "adjust_gain":            8.0,
            "balance_level":          5.0,  # continuous smoothed balancing
            "normalize_fader":      120.0,  # startup normalization (no signal needed)
            "mute_channel":          60.0,
            "unmute_channel":        60.0,
            "llm_recommendation":    30.0,
        }
        # Per-channel smoothed level adjustment tracking (for gradual balancing)
        self._level_smooth: Dict[int, float] = {}
        # Signal analyzer import (lazy, avoid import-time errors)
        self._signal_analyzer_available: Optional[bool] = None
        logger.info(
            "MixingAgent initialized mode=%s kb_first=%s",
            mode.value,
            kb_first,
        )

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
        if self.kb_first:
            actions = self._decide_kb_first(observations)
        else:
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

    # ── Instrument profiles ───────────────────────────────────────────────────

    # HPF cut-off frequencies (Hz) — applied once at startup
    _INST_HPF: Dict[str, float] = {
        "lead_vocal": 90.0,   "leadvocal":  90.0,
        "back_vocal": 100.0,  "backvocal":  100.0,  "backing_vocal": 100.0,
        "vocal":       90.0,
        "snare":      100.0,
        "hihat":      200.0,  "ride":       200.0,
        "overheads":  100.0,  "overhead":   100.0,
        "room":        80.0,
        "electricguitar": 80.0,  "electric_guitar": 80.0,
        "acousticguitar": 80.0,  "acoustic_guitar": 80.0,
        "accordion":  80.0,
    }

    # No HPF for these (bass-heavy fundamentals)
    _NO_HPF = frozenset({
        "kick", "bass", "bass_guitar", "synth_bass", "sub",
        "floor_tom", "tom", "playback", "room",
    })

    # EQ profiles — low_shelf / bands (1–4) / high_shelf
    _INST_EQ: Dict[str, dict] = {
        "kick": {
            "bands": [
                {"band": 1, "freq": 60.0,   "gain":  3.0, "q": 1.0},
                {"band": 2, "freq": 350.0,  "gain": -4.0, "q": 2.0},
                {"band": 3, "freq": 4000.0, "gain":  3.0, "q": 1.2},
            ],
            "high_shelf": {"gain": -2.0, "freq": 10000.0, "type": "SHV"},
        },
        "snare": {
            "low_shelf": {"gain": -2.0, "freq": 100.0, "type": "SHV"},
            "bands": [
                {"band": 1, "freq": 200.0,  "gain":  2.0, "q": 1.5},
                {"band": 2, "freq": 800.0,  "gain": -2.0, "q": 2.0},
                {"band": 3, "freq": 5000.0, "gain":  3.0, "q": 1.2},
            ],
        },
        "tom": {
            "bands": [
                {"band": 1, "freq": 100.0,  "gain":  2.0, "q": 1.0},
                {"band": 2, "freq": 400.0,  "gain": -3.0, "q": 2.0},
                {"band": 3, "freq": 3000.0, "gain":  2.0, "q": 1.5},
            ],
        },
        "hihat": {
            "bands": [
                {"band": 1, "freq": 500.0,  "gain": -2.0, "q": 2.0},
            ],
            "high_shelf": {"gain": 2.0, "freq": 10000.0, "type": "SHV"},
        },
        "ride": {
            "bands": [
                {"band": 1, "freq": 400.0,  "gain": -2.0, "q": 2.0},
            ],
            "high_shelf": {"gain": 1.5, "freq": 8000.0, "type": "SHV"},
        },
        "overheads": {
            "low_shelf": {"gain": -2.0, "freq": 200.0, "type": "SHV"},
            "high_shelf": {"gain": 1.5, "freq": 8000.0, "type": "SHV"},
        },
        "overhead": {
            "low_shelf": {"gain": -2.0, "freq": 200.0, "type": "SHV"},
            "high_shelf": {"gain": 1.5, "freq": 8000.0, "type": "SHV"},
        },
        "room": {
            "low_shelf": {"gain": -4.0, "freq": 200.0, "type": "SHV"},
            "bands": [
                {"band": 1, "freq": 500.0, "gain": -2.0, "q": 2.0},
            ],
        },
        "bass": {
            "low_shelf": {"gain": 2.0, "freq": 80.0, "type": "SHV"},
            "bands": [
                {"band": 1, "freq": 250.0, "gain": -2.0, "q": 1.5},
                {"band": 2, "freq": 700.0, "gain":  1.5, "q": 1.5},
            ],
        },
        "bass_guitar": {
            "low_shelf": {"gain": 2.0, "freq": 80.0, "type": "SHV"},
            "bands": [
                {"band": 1, "freq": 250.0, "gain": -2.0, "q": 1.5},
                {"band": 2, "freq": 700.0, "gain":  1.5, "q": 1.5},
            ],
        },
        "lead_vocal": {
            "bands": [
                {"band": 1, "freq":  250.0, "gain": -2.0, "q": 1.5},
                {"band": 2, "freq": 1000.0, "gain": -1.5, "q": 2.0},
                {"band": 3, "freq": 3000.0, "gain":  2.0, "q": 1.0},
            ],
            "high_shelf": {"gain": 1.5, "freq": 10000.0, "type": "SHV"},
        },
        "leadvocal": {
            "bands": [
                {"band": 1, "freq":  250.0, "gain": -2.0, "q": 1.5},
                {"band": 2, "freq": 1000.0, "gain": -1.5, "q": 2.0},
                {"band": 3, "freq": 3000.0, "gain":  2.0, "q": 1.0},
            ],
            "high_shelf": {"gain": 1.5, "freq": 10000.0, "type": "SHV"},
        },
        "back_vocal": {
            "bands": [
                {"band": 1, "freq":  250.0, "gain": -3.0, "q": 1.5},
                {"band": 2, "freq": 3000.0, "gain":  1.5, "q": 1.0},
            ],
            "high_shelf": {"gain": 1.0, "freq": 10000.0, "type": "SHV"},
        },
        "backvocal": {
            "bands": [
                {"band": 1, "freq":  250.0, "gain": -3.0, "q": 1.5},
                {"band": 2, "freq": 3000.0, "gain":  1.5, "q": 1.0},
            ],
            "high_shelf": {"gain": 1.0, "freq": 10000.0, "type": "SHV"},
        },
        "electricguitar": {
            "bands": [
                {"band": 1, "freq":  300.0, "gain": -2.0, "q": 2.0},
                {"band": 2, "freq": 2500.0, "gain":  2.0, "q": 1.2},
                {"band": 3, "freq": 5000.0, "gain": -1.5, "q": 1.5},
            ],
        },
        "electric_guitar": {
            "bands": [
                {"band": 1, "freq":  300.0, "gain": -2.0, "q": 2.0},
                {"band": 2, "freq": 2500.0, "gain":  2.0, "q": 1.2},
                {"band": 3, "freq": 5000.0, "gain": -1.5, "q": 1.5},
            ],
        },
        "accordion": {
            "bands": [
                {"band": 1, "freq":  300.0, "gain": -2.0, "q": 2.0},
                {"band": 2, "freq": 2000.0, "gain":  1.5, "q": 1.2},
            ],
        },
    }

    # Full compressor profiles: thr/ratio/attack/release/knee/makeup/det
    _INST_COMP_FULL: Dict[str, dict] = {
        "kick":          {"thr": -16.0, "ratio": 4.0, "att":  6.0, "rel": 100.0, "knee": 2, "makeup": 2.0, "det": "PEAK"},
        "snare":         {"thr": -16.0, "ratio": 4.0, "att":  5.0, "rel":  80.0, "knee": 1, "makeup": 1.0, "det": "PEAK"},
        "tom":           {"thr": -20.0, "ratio": 4.0, "att":  8.0, "rel": 120.0, "knee": 2, "makeup": 1.0, "det": "PEAK"},
        "lead_vocal":    {"thr": -18.0, "ratio": 3.0, "att": 10.0, "rel": 120.0, "knee": 2, "makeup": 2.0, "det": "RMS"},
        "leadvocal":     {"thr": -18.0, "ratio": 3.0, "att": 10.0, "rel": 120.0, "knee": 2, "makeup": 2.0, "det": "RMS"},
        "back_vocal":    {"thr": -20.0, "ratio": 3.0, "att": 12.0, "rel": 150.0, "knee": 2, "makeup": 2.0, "det": "RMS"},
        "backvocal":     {"thr": -20.0, "ratio": 3.0, "att": 12.0, "rel": 150.0, "knee": 2, "makeup": 2.0, "det": "RMS"},
        "bass":          {"thr": -20.0, "ratio": 3.5, "att": 30.0, "rel": 200.0, "knee": 3, "makeup": 2.0, "det": "RMS"},
        "bass_guitar":   {"thr": -20.0, "ratio": 3.5, "att": 30.0, "rel": 200.0, "knee": 3, "makeup": 2.0, "det": "RMS"},
        "electricguitar":{"thr": -22.0, "ratio": 3.0, "att": 20.0, "rel": 150.0, "knee": 2, "makeup": 1.0, "det": "RMS"},
        "electric_guitar":{"thr":-22.0, "ratio": 3.0, "att": 20.0, "rel": 150.0, "knee": 2, "makeup": 1.0, "det": "RMS"},
        "accordion":     {"thr": -22.0, "ratio": 2.5, "att": 25.0, "rel": 200.0, "knee": 2, "makeup": 1.0, "det": "RMS"},
        "overheads":     {"thr": -28.0, "ratio": 2.0, "att": 20.0, "rel": 300.0, "knee": 3, "makeup": 0.0, "det": "RMS"},
        "overhead":      {"thr": -28.0, "ratio": 2.0, "att": 20.0, "rel": 300.0, "knee": 3, "makeup": 0.0, "det": "RMS"},
    }

    # Target LUFS (momentary) per instrument for level balancing
    _TARGET_LUFS: Dict[str, float] = {
        "kick":          -20.0,
        "snare":         -20.0,
        "tom":           -24.0,
        "hihat":         -28.0,
        "ride":          -30.0,
        "overheads":     -24.0,
        "overhead":      -24.0,
        "room":          -32.0,
        "bass":          -18.0,
        "bass_guitar":   -18.0,
        "lead_vocal":    -16.0,
        "leadvocal":     -16.0,
        "back_vocal":    -22.0,
        "backvocal":     -22.0,
        "backing_vocal": -22.0,
        "electricguitar":-22.0,
        "electric_guitar":-22.0,
        "accordion":     -24.0,
        "playback":      -20.0,
    }

    # Nominal fader positions (dB) by instrument type for startup normalization.
    # Applied once per session when no audio signal is present, to put channels
    # at reasonable starting positions.
    _NOMINAL_FADER: Dict[str, float] = {
        "kick":           0.0,
        "snare":          0.0,
        "tom":           -3.0,
        "hihat":         -6.0,
        "ride":          -6.0,
        "overheads":     -6.0,
        "overhead":      -6.0,
        "room":         -10.0,
        "bass":           0.0,
        "bass_guitar":    0.0,
        "lead_vocal":    +3.0,
        "leadvocal":     +3.0,
        "back_vocal":    -3.0,
        "backvocal":     -3.0,
        "backing_vocal": -3.0,
        "electricguitar": -3.0,
        "electric_guitar":-3.0,
        "accordion":     -3.0,
        "playback":      -6.0,
    }

    def _on_cooldown(self, ch: int, action_type: str, now: float) -> bool:
        """Return True if this (channel, action_type) is still cooling down."""
        key = (ch, action_type)
        last = self._last_applied.get(key, 0.0)
        return (now - last) < self._cooldowns.get(action_type, 30.0)

    def _mark_applied(self, ch: int, action_type: str, now: float) -> None:
        self._last_applied[(ch, action_type)] = now

    def _load_signal_analyzer(self):
        """Lazy-import signal_analyzer. Returns module or None."""
        if self._signal_analyzer_available is False:
            return None
        try:
            from ai import signal_analyzer as sa
            self._signal_analyzer_available = True
            return sa
        except ImportError:
            try:
                import signal_analyzer as sa
                self._signal_analyzer_available = True
                return sa
            except ImportError:
                self._signal_analyzer_available = False
                return None

    def _decide_kb_first(self, observations: Dict) -> List[AgentAction]:
        """Analysis-driven decisions — measure signal first, then decide corrections.

        Decision pipeline per channel:
        1. SAFETY GUARD    — clip/hot level → immediate fader trim
        2. HPF             — instrument HPF freq (structural, one-shot)
        3. EQ (analysis)   — compare measured spectral shape vs KB target, apply only needed bands
        4. COMPRESSOR      — initial params from KB; then adjust based on crest factor analysis
        5. COMPRESSION MON — ongoing: monitor crest factor, adjust threshold if over/under
        6. LEVEL BALANCE   — smooth gradual LUFS tracking (±0.5 dB/step, 5s cooldown)
        7. PHASE CORRECTION— apply delay/polarity correction from snare bleed analysis
        8. LLM fallback    — for attention channels when LLM is configured
        """
        sa = self._load_signal_analyzer()
        actions: List[AgentAction] = []
        now = time.time()
        llm_used_this_cycle = False

        for ch_id, ch_state in observations.get("channels", {}).items():
            ch = int(ch_id)
            raw_inst = (ch_state.get("instrument") or "unknown")
            inst = raw_inst.lower()
            inst_key = inst.replace(" ", "_").replace("-", "_")

            peak = float(ch_state.get("peak_db", ch_state.get("true_peak_db", -100.0)))
            lufs_m = float(ch_state.get("lufs_momentary", ch_state.get("lufs", -100.0)))
            rms_db = float(ch_state.get("rms_db", lufs_m))
            crest_factor = float(ch_state.get("crest_factor_db", max(0.0, peak - rms_db)))
            is_muted = bool(ch_state.get("is_muted", False))
            has_signal = peak > -60.0
            band_energy: Dict[str, float] = ch_state.get("band_energy") or {}
            comp_thr = float(ch_state.get("comp_threshold_db", -18.0))

            # ── 1. SAFETY: clip/hot level → immediate fader trim ─────────
            hot = peak > -3.0 or lufs_m > -9.0
            very_hot = peak > 0.0 or lufs_m > -6.0
            if hot and not is_muted and not self._on_cooldown(ch, "reduce_gain", now):
                amount = -3.0 if very_hot else -1.5
                kb_title = ""
                if self.kb:
                    hits = self.kb.search(
                        f"{inst} gain staging headroom clipping fader", n_results=2
                    )
                    kb_title = hits[0].metadata.get("title", "") if hits else ""
                actions.append(AgentAction(
                    action_type="reduce_gain",
                    channel=ch,
                    parameters={"amount_db": amount, "channel": ch},
                    priority=1,
                    confidence=0.92,
                    reason=(
                        f"{'CLIP' if very_hot else 'Hot'} level "
                        f"peak={peak:.1f}dB LUFS={lufs_m:.1f} → {amount:+.1f}dB; {kb_title[:50]}"
                    ),
                    source="analysis",
                    timestamp=now,
                ))

            # ── 2. HPF — structural, once per session ────────────────────
            hpf_freq = self._INST_HPF.get(inst_key) or self._INST_HPF.get(inst)
            if hpf_freq is None and "vocal" in inst:
                hpf_freq = 90.0
            if hpf_freq is None and ("guitar" in inst or "gtr" in inst):
                hpf_freq = 80.0

            if hpf_freq and inst_key not in self._NO_HPF and inst not in self._NO_HPF:
                if not self._on_cooldown(ch, "apply_hpf", now):
                    # If sub band is weak in measured spectrum, confirm HPF is appropriate
                    sub_energy = band_energy.get("sub", -100.0)
                    hpf_needed = sub_energy > -60.0  # sub energy present → HPF needed
                    if not has_signal:
                        hpf_needed = True  # No audio yet — apply preemptively
                    if hpf_needed:
                        kb_reason = f"Remove sub-rumble below {hpf_freq:.0f}Hz"
                        if has_signal and sub_energy > -40.0:
                            kb_reason = (
                                f"Sub energy {sub_energy:.1f}dB detected below {hpf_freq:.0f}Hz → cut"
                            )
                        actions.append(AgentAction(
                            action_type="apply_hpf",
                            channel=ch,
                            parameters={"frequency": hpf_freq, "channel": ch},
                            priority=3,
                            confidence=0.88,
                            reason=f"HPF {hpf_freq:.0f}Hz for {raw_inst}: {kb_reason}",
                            source="analysis",
                            timestamp=now,
                        ))

            # ── 3. EQ — analysis-driven, re-check every 45s ─────────────
            if not self._on_cooldown(ch, "apply_eq", now):
                inst_target = (
                    sa.INST_BAND_TARGETS.get(inst_key) if sa else None
                ) or (sa.INST_BAND_TARGETS.get(inst) if sa else None)

                if inst_target and has_signal and band_energy:
                    # Analysis path: compare measured spectrum vs target
                    corrections = sa.compute_eq_corrections(
                        measured=band_energy,
                        inst_target=inst_target,
                        min_correction_db=1.8,
                        max_correction_db=4.0,
                        max_corrections=3,
                    )
                    if corrections:
                        # Build EQ profile from analysis results
                        eq_profile: Dict = {"bands": [], "low_shelf": None, "high_shelf": None}
                        band_slot_counter = 1
                        for corr in corrections:
                            slot = corr.get("band_slot", "band")
                            if slot == "low_shelf":
                                eq_profile["low_shelf"] = {
                                    "gain": corr["gain"], "freq": corr["freq"], "type": "SHV"
                                }
                            elif slot == "high_shelf":
                                eq_profile["high_shelf"] = {
                                    "gain": corr["gain"], "freq": corr["freq"], "type": "SHV"
                                }
                            else:
                                if band_slot_counter <= 4:
                                    eq_profile["bands"].append({
                                        "band": band_slot_counter,
                                        "freq": corr["freq"],
                                        "gain": corr["gain"],
                                        "q": corr.get("q", 1.5),
                                    })
                                    band_slot_counter += 1

                        desc = " | ".join(
                            f"{c['gain']:+.1f}dB@{c['freq']:.0f}Hz ({c['band']})"
                            for c in corrections
                        )
                        actions.append(AgentAction(
                            action_type="apply_eq",
                            channel=ch,
                            parameters={"profile": eq_profile, "channel": ch},
                            priority=4,
                            confidence=0.82,
                            reason=f"Spectrum analysis {raw_inst}: {desc}",
                            source="analysis",
                            timestamp=now,
                        ))
                elif not has_signal:
                    # No signal yet: apply knowledge-profile EQ preemptively
                    eq_profile = self._INST_EQ.get(inst_key) or self._INST_EQ.get(inst)
                    if eq_profile:
                        bands_desc = " | ".join(
                            f"{bd.get('gain',0):+.0f}dB@{bd.get('freq',0):.0f}Hz"
                            for bd in eq_profile.get("bands", [])
                        )
                        actions.append(AgentAction(
                            action_type="apply_eq",
                            channel=ch,
                            parameters={"profile": eq_profile, "channel": ch},
                            priority=4,
                            confidence=0.75,
                            reason=f"KB profile EQ (no signal yet) {raw_inst}: {bands_desc}",
                            source="knowledge_base",
                            timestamp=now,
                        ))

            # ── 4. COMPRESSOR — initial setup from KB, then monitoring ───
            comp = self._INST_COMP_FULL.get(inst_key) or self._INST_COMP_FULL.get(inst)
            if comp and not self._on_cooldown(ch, "adjust_compressor", now):
                actions.append(AgentAction(
                    action_type="adjust_compressor",
                    channel=ch,
                    parameters={
                        "threshold": comp["thr"],
                        "ratio":     comp["ratio"],
                        "attack":    comp["att"],
                        "release":   comp["rel"],
                        "knee":      comp.get("knee", 2),
                        "makeup":    comp.get("makeup", 0.0),
                        "det":       comp.get("det", "RMS"),
                        "channel":   ch,
                    },
                    priority=5,
                    confidence=0.85,
                    reason=(
                        f"Comp initial: {comp['thr']:.0f}dBFS/{comp['ratio']:.1f}:1 "
                        f"att={comp['att']:.0f}ms rel={comp['rel']:.0f}ms "
                        f"{comp.get('det','RMS')}"
                    ),
                    source="knowledge_base",
                    timestamp=now,
                ))

            # ── 5. COMPRESSION MONITORING — crest factor analysis ────────
            if (
                comp
                and has_signal
                and not self._on_cooldown(ch, "adjust_compression", now)
                and sa is not None
            ):
                comp_state = sa.analyze_compression_state(
                    peak_db=peak,
                    rms_db=rms_db,
                    crest_factor_db=crest_factor,
                    current_threshold_db=comp_thr,
                    inst_type=inst,
                )
                if comp_state["state"] != "ok" and comp_state["confidence"] > 0.6:
                    delta = comp_state["threshold_delta"]
                    new_thr = max(-50.0, min(-6.0, comp_thr + delta))
                    ideal_range = comp_state.get("ideal_range", (6, 15))
                    actions.append(AgentAction(
                        action_type="adjust_compression",
                        channel=ch,
                        parameters={
                            "threshold": new_thr,
                            "delta": delta,
                            "channel": ch,
                        },
                        priority=5,
                        confidence=comp_state["confidence"],
                        reason=(
                            f"Comp {comp_state['state']}: crest={crest_factor:.1f}dB "
                            f"(ideal {ideal_range[0]}-{ideal_range[1]}dB) → thr {delta:+.1f}dB"
                        ),
                        source="analysis",
                        timestamp=now,
                    ))

            # ── 6. LEVEL BALANCE — smooth gradual LUFS tracking ──────────
            # Works with AudioCapture LUFS (precise) or Wing meter LUFS estimate (fallback).
            target_lufs = self._TARGET_LUFS.get(inst_key) or self._TARGET_LUFS.get(inst)
            effective_lufs = lufs_m if lufs_m > -90.0 else None
            max_fader_db = float(ch_state.get("max_fader_db", 0.0))
            current_fdr = float(ch_state.get("current_fader_db", 0.0))
            # Skip upward balance if fader is already at the safety ceiling
            at_fader_ceiling = current_fdr >= (max_fader_db - 0.1)
            if (
                target_lufs is not None
                and effective_lufs is not None
                and has_signal
                and not is_muted
                and effective_lufs > -55.0
                and not self._on_cooldown(ch, "balance_level", now)
                and sa is not None
            ):
                prev_adj = self._level_smooth.get(ch, 0.0)
                adjust = sa.smooth_level_adjustment(
                    current_lufs=effective_lufs,
                    target_lufs=target_lufs,
                    prev_adjustment=prev_adj,
                    alpha=0.4,
                    max_step_db=0.5,
                    deadband_db=0.8,  # Tighter deadband → more responsive corrections
                )
                # Skip upward moves when fader is already at the safety ceiling
                if adjust > 0 and at_fader_ceiling:
                    adjust = 0.0
                if adjust != 0.0:
                    self._level_smooth[ch] = adjust
                    direction = "↓" if adjust < 0 else "↑"
                    actions.append(AgentAction(
                        action_type="balance_level",
                        channel=ch,
                        parameters={"amount_db": adjust, "channel": ch,
                                    "target_lufs": target_lufs,
                                    "current_lufs": effective_lufs},
                        priority=2,
                        confidence=0.82,
                        reason=(
                            f"Balance {raw_inst}: {effective_lufs:.1f}→{target_lufs:.1f}LUFS "
                            f"{direction}{abs(adjust):.2f}dB"
                        ),
                        source="analysis",
                        timestamp=now,
                    ))

            # ── 6b. FADER NORMALIZATION — startup positioning without audio ──
            # When there's no audio signal data, nudge the fader toward the
            # nominal position for the instrument type. This gives useful
            # channel setup even before Dante/audio capture is routing signal.
            if (
                not has_signal
                and not is_muted
                and inst_key in self._NOMINAL_FADER
                and not self._on_cooldown(ch, "normalize_fader", now)
            ):
                nominal = min(self._NOMINAL_FADER[inst_key], max_fader_db)
                # current_fdr already defined in level balance section above
                delta = nominal - current_fdr
                if abs(delta) > 0.5:  # Only if more than 0.5 dB away
                    step = max(-2.0, min(2.0, delta * 0.3))  # 30% correction, ±2dB cap
                    actions.append(AgentAction(
                        action_type="normalize_fader",
                        channel=ch,
                        parameters={"amount_db": round(step, 2), "channel": ch,
                                    "nominal_db": nominal, "current_fdr": current_fdr},
                        priority=3,
                        confidence=0.65,
                        reason=(
                            f"Normalize {raw_inst}: fader {current_fdr:.1f}→{nominal:.1f}dB "
                            f"({step:+.2f}dB, no Dante signal)"
                        ),
                        source="analysis",
                        timestamp=now,
                    ))

            # ── 7. PHASE CORRECTION — snare bleed analysis results ───────
            phase_delay = float(ch_state.get("phase_delay_ms", 0.0))
            phase_strength = float(ch_state.get("phase_strength", 0.0))
            phase_inverted = bool(ch_state.get("phase_inverted", False))
            if (
                phase_strength > 0.20
                and abs(phase_delay) > 0.3  # >0.3ms is audible
                and not self._on_cooldown(ch, "apply_phase_correction", now)
            ):
                actions.append(AgentAction(
                    action_type="apply_phase_correction",
                    channel=ch,
                    parameters={
                        "delay_ms": phase_delay,
                        "invert": phase_inverted,
                        "channel": ch,
                    },
                    priority=3,
                    confidence=min(0.95, 0.5 + phase_strength * 0.5),
                    reason=(
                        f"Snare bleed in {raw_inst}: delay={phase_delay:.2f}ms "
                        f"corr={phase_strength:.2f} {'INVERT' if phase_inverted else ''}"
                    ),
                    source="analysis",
                    timestamp=now,
                ))

            # ── 8. LLM fallback ──────────────────────────────────────────
            if (
                self.llm
                and not llm_used_this_cycle
                and ch_state.get("needs_attention", False)
                and now - self._last_llm_query > self._llm_query_interval
            ):
                self._last_llm_query = now
                llm_used_this_cycle = True
                try:
                    ctx: List[str] = []
                    if self.kb:
                        ctx = [
                            e.content
                            for e in self.kb.search(f"{inst} live mix problem", n_results=3)
                        ]
                    rec = self.llm.get_mix_recommendation(ch_state, ctx)
                    if rec and rec.get("gain_db") is not None:
                        actions.append(AgentAction(
                            action_type="llm_recommendation",
                            channel=ch,
                            parameters=rec,
                            priority=3,
                            confidence=float(rec.get("confidence", 0.55) or 0.55),
                            reason=rec.get("reason", "LLM recommendation"),
                            source="llm",
                            timestamp=now,
                        ))
                except Exception as e:
                    logger.debug("LLM consultation error: %s", e)

        actions.sort(key=lambda a: (a.priority, -a.confidence))
        return actions[: self._max_actions_per_cycle]

    async def _act(self, actions: List[AgentAction]):
        """Execute decided actions on the mixer."""
        for action in actions:
            try:
                applied = False
                if self.mixer:
                    ch = int(action.parameters.get("channel", action.channel))
                    if action.action_type == 'reduce_gain':
                        delta = float(action.parameters.get('amount_db', -3.0))
                        if hasattr(self.mixer, "adjust_channel_fader"):
                            self.mixer.adjust_channel_fader(ch, delta)
                        else:
                            cur = getattr(
                                self.mixer, "get_channel_fader", lambda _c: -20.0
                            )(ch)
                            self.mixer.set_channel_fader(ch, float(cur) + delta)
                        applied = True
                    elif action.action_type == 'adjust_gain':
                        adj = action.parameters.get('adjustment_db')
                        if adj is None:
                            adj = action.parameters.get('target_relative_db', 1.0)
                        adj = float(adj)
                        if hasattr(self.mixer, "adjust_channel_fader"):
                            self.mixer.adjust_channel_fader(ch, adj)
                        else:
                            cur = getattr(
                                self.mixer, "get_channel_fader", lambda _c: -20.0
                            )(ch)
                            self.mixer.set_channel_fader(ch, float(cur) + adj)
                        applied = True
                    elif action.action_type == 'mute_channel':
                        self.mixer.set_channel_mute(ch, 1)
                        applied = True
                    elif action.action_type == 'unmute_channel':
                        self.mixer.set_channel_mute(ch, 0)
                        applied = True
                    elif action.action_type == 'apply_hpf':
                        freq = action.parameters.get('frequency', 80)
                        self.mixer.set_channel_hpf(ch, float(freq))
                        applied = True

                    elif action.action_type == 'apply_eq':
                        profile = action.parameters.get('profile') or {}
                        if hasattr(self.mixer, 'apply_eq_profile'):
                            self.mixer.apply_eq_profile(ch, profile)
                        else:
                            # Fallback: individual calls
                            self.mixer.set_eq_on(ch, 1)
                            for bd in profile.get('bands', []):
                                band_n = int(bd.get('band', 1))
                                if 1 <= band_n <= 4:
                                    self.mixer.set_eq_band(
                                        ch, band_n,
                                        freq=float(bd['freq']) if 'freq' in bd else None,
                                        gain=float(bd['gain']) if 'gain' in bd else None,
                                        q=float(bd.get('q', 1.5)),
                                    )
                        applied = True

                    elif action.action_type == 'adjust_compressor':
                        p = action.parameters
                        if hasattr(self.mixer, 'set_channel_compressor_full'):
                            self.mixer.set_channel_compressor_full(
                                ch,
                                threshold=float(p.get('threshold', p.get('threshold_db', -18.0))),
                                ratio=float(p.get('ratio', 3.0)),
                                attack=float(p.get('attack', p.get('attack_ms', 10.0))),
                                release=float(p.get('release', p.get('release_ms', 100.0))),
                                knee=int(p.get('knee', 2)),
                                makeup=float(p.get('makeup', 0.0)),
                                det=str(p.get('det', 'RMS')),
                            )
                        else:
                            self.mixer.set_channel_compressor(
                                ch,
                                threshold=float(p.get('threshold', p.get('threshold_db', -18.0))),
                                ratio=float(p.get('ratio', 3.0)),
                                attack=float(p.get('attack', p.get('attack_ms', 10.0))),
                                release=float(p.get('release', p.get('release_ms', 100.0))),
                            )
                        applied = True

                    elif action.action_type in ('balance_level', 'normalize_fader'):
                        amount = float(action.parameters.get('amount_db', 0.0))
                        if hasattr(self.mixer, 'adjust_channel_fader'):
                            self.mixer.adjust_channel_fader(ch, amount)
                        applied = True

                    elif action.action_type == 'apply_phase_correction':
                        delay_ms = float(action.parameters.get('delay_ms', 0.0))
                        invert = bool(action.parameters.get('invert', False))
                        if hasattr(self.mixer, 'set_channel_delay') and abs(delay_ms) > 0.1:
                            self.mixer.set_channel_delay(ch, abs(delay_ms), mode="MS")
                        if hasattr(self.mixer, 'set_channel_phase_invert'):
                            self.mixer.set_channel_phase_invert(ch, 1 if invert else 0)
                        applied = True

                    elif action.action_type == 'adjust_compression':
                        # Fine-tune compressor threshold based on crest-factor analysis
                        new_thr = float(action.parameters.get('threshold', -18.0))
                        if hasattr(self.mixer, 'set_compressor_threshold'):
                            self.mixer.set_compressor_threshold(ch, new_thr)
                        applied = True

                    elif action.action_type == 'adjust_deesser':
                        self.mixer.set_channel_deesser(
                            ch,
                            frequency=action.parameters.get('frequency', 6500),
                            threshold_db=action.parameters.get('threshold_db', -20),
                            ratio=action.parameters.get('ratio', 4.0),
                        )
                        applied = True

                    elif action.action_type == 'llm_recommendation':
                        gdb = action.parameters.get("gain_db")
                        if gdb is not None and hasattr(
                            self.mixer, "adjust_channel_fader"
                        ):
                            self.mixer.adjust_channel_fader(ch, float(gdb))
                            applied = True

                if applied:
                    self._mark_applied(action.channel, action.action_type, time.time())
                    self._action_history.append(action)
                    self.state.applied_actions.append(action)

                    # Trim history to prevent memory growth
                    if len(self._action_history) > 1000:
                        self._action_history = self._action_history[-500:]
                    if len(self.state.applied_actions) > 100:
                        self.state.applied_actions = self.state.applied_actions[-50:]

                    logger.info(
                        "Applied %s ch%s conf=%.2f | %s",
                        action.action_type, action.channel,
                        action.confidence, action.reason[:100],
                    )
            except Exception as e:
                error_msg = f"Action error: {action.action_type} ch{action.channel}: {e}"
                logger.error(error_msg)
                self.state.errors.append(error_msg)

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
