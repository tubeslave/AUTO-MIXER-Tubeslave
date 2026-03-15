"""
AIAgent — orchestrator for the AUTO MIXER Tubeslave AI system.

Implements a 3-tier decision architecture:
1. Rule-based (fast, deterministic) via RuleEngine
2. Local LLM (Ollama) for nuanced decisions
3. Cloud LLM (Perplexity) as final fallback

Exposes 11 function-calling tools for mixer control and provides
a run_soundcheck() method for automated channel setup.
"""

import asyncio
import json
import logging
import math
import os
import re
import time
import threading
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from .knowledge_base import KnowledgeBase
from .rule_engine import RuleEngine
from .llm_client import OllamaClient, PerplexityClient, FallbackChain

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety limits
# ---------------------------------------------------------------------------

MAX_GAIN_CHANGE_PER_STEP_DB = 6.0    # Max gain change in a single operation
MAX_FADER_DB = 10.0                   # Absolute max fader value (Wing max is +10)
MIN_FADER_DB = -144.0                 # Absolute min fader value (Wing -inf)
FEEDBACK_OVERRIDE_GAIN_DB = -12.0     # Emergency cut on feedback detection
MAX_EQ_BOOST_DB = 12.0               # Max EQ boost allowed
MAX_EQ_CUT_DB = -15.0                # Max EQ cut allowed
SOUNDCHECK_SETTLE_TIME = 0.5         # Seconds to wait between channel adjustments


# ---------------------------------------------------------------------------
# Tool definitions for function calling
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "set_channel_fader",
        "description": "Set channel fader level in dB (-144 to +10)",
        "parameters": {"ch": "int", "value": "float"},
    },
    {
        "name": "set_channel_eq",
        "description": "Set a single EQ band on a channel",
        "parameters": {
            "ch": "int", "band": "int",
            "freq": "float", "gain": "float", "q": "float",
        },
    },
    {
        "name": "set_channel_compressor",
        "description": "Set compressor parameters on a channel",
        "parameters": {
            "ch": "int", "threshold": "float", "ratio": "float",
            "attack": "float", "release": "float",
        },
    },
    {
        "name": "set_channel_gate",
        "description": "Set gate parameters on a channel",
        "parameters": {
            "ch": "int", "threshold": "float", "attack": "float",
            "hold": "float", "release": "float",
        },
    },
    {
        "name": "set_channel_pan",
        "description": "Set channel pan position (-100 to +100)",
        "parameters": {"ch": "int", "value": "float"},
    },
    {
        "name": "set_channel_mute",
        "description": "Mute or unmute a channel",
        "parameters": {"ch": "int", "muted": "bool"},
    },
    {
        "name": "set_channel_hpf",
        "description": "Set high-pass filter frequency on a channel",
        "parameters": {"ch": "int", "frequency": "float"},
    },
    {
        "name": "get_channel_levels",
        "description": "Get current levels for a channel",
        "parameters": {"ch": "int"},
    },
    {
        "name": "get_mix_quality",
        "description": "Get overall mix quality metrics",
        "parameters": {},
    },
    {
        "name": "search_knowledge",
        "description": "Search the mixing knowledge base",
        "parameters": {"query": "str"},
    },
    {
        "name": "apply_preset",
        "description": "Apply a full instrument preset to a channel",
        "parameters": {"ch": "int", "preset_name": "str"},
    },
]


# ---------------------------------------------------------------------------
# AIAgent
# ---------------------------------------------------------------------------

class AIAgent:
    """
    AI mixing agent with 3-tier routing and 11 function-calling tools.

    Tier 1: Rule engine (instant, deterministic)
    Tier 2: Local LLM via Ollama (complex reasoning, private)
    Tier 3: Cloud LLM via Perplexity (search-grounded, last resort)
    """

    def __init__(
        self,
        wing_client: Optional[Any] = None,
        knowledge_dir: Optional[str] = None,
        ollama_model: str = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434",
        perplexity_api_key: Optional[str] = None,
    ):
        """
        Initialize the AI agent.

        Args:
            wing_client: WingClient or EnhancedOSCClient instance for mixer control.
                         If None, tool calls will be logged but not sent.
            knowledge_dir: Path to knowledge markdown files directory.
                           Defaults to backend/ai/knowledge/.
            ollama_model: Ollama model to use.
            ollama_url: Ollama server URL.
            perplexity_api_key: Perplexity API key (or set PERPLEXITY_API_KEY env var).
        """
        self._wing = wing_client
        self._lock = threading.Lock()

        # Tier 1: Rule engine
        self.rule_engine = RuleEngine()

        # Knowledge base
        if knowledge_dir is None:
            knowledge_dir = str(
                Path(__file__).parent / "knowledge"
            )
        self.knowledge_base = KnowledgeBase()
        self._index_knowledge(knowledge_dir)

        # Tier 2: Local LLM
        self._ollama = OllamaClient(
            base_url=ollama_url,
            model=ollama_model,
        )

        # Tier 3: Cloud LLM
        self._perplexity = PerplexityClient(
            api_key=perplexity_api_key,
        )

        # Fallback chain: Ollama first, then Perplexity
        self._llm_chain = FallbackChain(
            clients=[self._ollama, self._perplexity],
            timeout_per_client=60.0,
        )

        # Channel state cache
        self._channel_state: Dict[int, Dict[str, Any]] = {}
        # Tracks instrument type per channel
        self._channel_instruments: Dict[int, str] = {}

        # Tool registry
        self._tools: Dict[str, Callable[..., Coroutine]] = {
            "set_channel_fader": self.set_channel_fader,
            "set_channel_eq": self.set_channel_eq,
            "set_channel_compressor": self.set_channel_compressor,
            "set_channel_gate": self.set_channel_gate,
            "set_channel_pan": self.set_channel_pan,
            "set_channel_mute": self.set_channel_mute,
            "set_channel_hpf": self.set_channel_hpf,
            "get_channel_levels": self.get_channel_levels,
            "get_mix_quality": self.get_mix_quality,
            "search_knowledge": self.search_knowledge,
            "apply_preset": self.apply_preset,
        }

        logger.info(
            f"AIAgent initialized (knowledge={self.knowledge_base.document_count} docs, "
            f"backend={self.knowledge_base.backend_name})"
        )

    def _index_knowledge(self, directory: str) -> None:
        """Index knowledge files from the given directory."""
        if os.path.isdir(directory):
            count = self.knowledge_base.index_all(directory)
            logger.info(f"Indexed {count} knowledge chunks from {directory}")
        else:
            logger.warning(f"Knowledge directory not found: {directory}")

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_fader(value: float) -> float:
        """Clamp fader value to safe range."""
        return max(MIN_FADER_DB, min(MAX_FADER_DB, value))

    @staticmethod
    def _clamp_eq_gain(gain: float) -> float:
        """Clamp EQ gain to safe range."""
        return max(MAX_EQ_CUT_DB, min(MAX_EQ_BOOST_DB, gain))

    def _enforce_gain_limit(self, ch: int, new_value: float) -> float:
        """
        Enforce maximum gain change per step.

        If the requested change exceeds MAX_GAIN_CHANGE_PER_STEP_DB,
        the value is limited to the maximum allowed step.
        """
        current = self._channel_state.get(ch, {}).get("fader", 0.0)
        delta = new_value - current
        if abs(delta) > MAX_GAIN_CHANGE_PER_STEP_DB:
            limited = current + (
                MAX_GAIN_CHANGE_PER_STEP_DB if delta > 0
                else -MAX_GAIN_CHANGE_PER_STEP_DB
            )
            logger.warning(
                f"Ch {ch}: gain change {delta:+.1f} dB exceeds limit, "
                f"clamped to {limited:.1f} dB (step limit: "
                f"{MAX_GAIN_CHANGE_PER_STEP_DB} dB)"
            )
            return limited
        return new_value

    # ------------------------------------------------------------------
    # 11 Function-calling tools
    # ------------------------------------------------------------------

    async def set_channel_fader(self, ch: int, value: float) -> Dict[str, Any]:
        """
        Set channel fader level.

        Args:
            ch: Channel number (1-40).
            value: Fader level in dB (-144 to +10).

        Returns:
            Result dict with applied value.
        """
        value = self._clamp_fader(value)
        value = self._enforce_gain_limit(ch, value)

        if self._wing is not None:
            try:
                address = f"/ch/{ch}/fdr"
                self._wing.send_float(address, value)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} fader: {exc}")
                return {"success": False, "error": str(exc)}

        self._channel_state.setdefault(ch, {})["fader"] = value
        logger.info(f"Ch {ch}: fader -> {value:.1f} dB")
        return {"success": True, "ch": ch, "fader": value}

    async def set_channel_eq(
        self, ch: int, band: int, freq: float, gain: float, q: float,
    ) -> Dict[str, Any]:
        """
        Set a single EQ band on a channel.

        Args:
            ch: Channel number (1-40).
            band: EQ band number (1-4 for parametric, 0=low shelf, 5=high shelf).
            freq: Center frequency in Hz (20-20000).
            gain: Gain in dB (-15 to +15).
            q: Q factor (0.44 to 10).

        Returns:
            Result dict with applied parameters.
        """
        gain = self._clamp_eq_gain(gain)
        freq = max(20.0, min(20000.0, freq))
        q = max(0.44, min(10.0, q))

        if self._wing is not None:
            try:
                # Map band number to Wing EQ addresses
                band_map = {0: "l", 1: "1", 2: "2", 3: "3", 4: "4", 5: "h"}
                band_key = band_map.get(band, str(band))
                self._wing.send_float(f"/ch/{ch}/eq/{band_key}f", freq)
                self._wing.send_float(f"/ch/{ch}/eq/{band_key}g", gain)
                self._wing.send_float(f"/ch/{ch}/eq/{band_key}q", q)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} EQ band {band}: {exc}")
                return {"success": False, "error": str(exc)}

        eq_state = self._channel_state.setdefault(ch, {}).setdefault("eq", {})
        eq_state[band] = {"freq": freq, "gain": gain, "q": q}
        logger.info(f"Ch {ch}: EQ band {band} -> {freq:.0f} Hz, {gain:+.1f} dB, Q={q:.2f}")
        return {"success": True, "ch": ch, "band": band, "freq": freq, "gain": gain, "q": q}

    async def set_channel_compressor(
        self, ch: int, threshold: float, ratio: float,
        attack: float, release: float,
    ) -> Dict[str, Any]:
        """
        Set compressor parameters on a channel.

        Args:
            ch: Channel number (1-40).
            threshold: Threshold in dB (-60 to 0).
            ratio: Compression ratio (1.1 to 100).
            attack: Attack time in ms (0 to 120).
            release: Release time in ms (4 to 4000).

        Returns:
            Result dict.
        """
        threshold = max(-60.0, min(0.0, threshold))
        ratio = max(1.1, min(100.0, ratio))
        attack = max(0.0, min(120.0, attack))
        release = max(4.0, min(4000.0, release))

        if self._wing is not None:
            try:
                self._wing.send_float(f"/ch/{ch}/dyn/thr", threshold)
                self._wing.send_float(f"/ch/{ch}/dyn/ratio", ratio)
                self._wing.send_float(f"/ch/{ch}/dyn/att", attack)
                self._wing.send_float(f"/ch/{ch}/dyn/rel", release)
                # Enable compressor
                self._wing.send_int(f"/ch/{ch}/dyn/on", 1)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} compressor: {exc}")
                return {"success": False, "error": str(exc)}

        comp_state = self._channel_state.setdefault(ch, {})
        comp_state["compressor"] = {
            "threshold": threshold, "ratio": ratio,
            "attack": attack, "release": release,
        }
        logger.info(
            f"Ch {ch}: comp -> thr={threshold:.1f} dB, ratio={ratio:.1f}:1, "
            f"att={attack:.1f} ms, rel={release:.1f} ms"
        )
        return {
            "success": True, "ch": ch,
            "threshold": threshold, "ratio": ratio,
            "attack": attack, "release": release,
        }

    async def set_channel_gate(
        self, ch: int, threshold: float, attack: float,
        hold: float, release: float,
    ) -> Dict[str, Any]:
        """
        Set gate parameters on a channel.

        Args:
            ch: Channel number (1-40).
            threshold: Gate threshold in dB (-80 to 0).
            attack: Attack time in ms (0 to 120).
            hold: Hold time in ms (0 to 200).
            release: Release time in ms (4 to 4000).

        Returns:
            Result dict.
        """
        threshold = max(-80.0, min(0.0, threshold))
        attack = max(0.0, min(120.0, attack))
        hold = max(0.0, min(200.0, hold))
        release = max(4.0, min(4000.0, release))

        if self._wing is not None:
            try:
                self._wing.send_float(f"/ch/{ch}/gate/thr", threshold)
                self._wing.send_float(f"/ch/{ch}/gate/att", attack)
                self._wing.send_float(f"/ch/{ch}/gate/hld", hold)
                self._wing.send_float(f"/ch/{ch}/gate/rel", release)
                self._wing.send_int(f"/ch/{ch}/gate/on", 1)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} gate: {exc}")
                return {"success": False, "error": str(exc)}

        gate_state = self._channel_state.setdefault(ch, {})
        gate_state["gate"] = {
            "threshold": threshold, "attack": attack,
            "hold": hold, "release": release,
        }
        logger.info(
            f"Ch {ch}: gate -> thr={threshold:.1f} dB, att={attack:.1f} ms, "
            f"hold={hold:.1f} ms, rel={release:.1f} ms"
        )
        return {
            "success": True, "ch": ch,
            "threshold": threshold, "attack": attack,
            "hold": hold, "release": release,
        }

    async def set_channel_pan(self, ch: int, value: float) -> Dict[str, Any]:
        """
        Set channel pan position.

        Args:
            ch: Channel number (1-40).
            value: Pan value (-100 = hard left, 0 = center, +100 = hard right).

        Returns:
            Result dict.
        """
        value = max(-100.0, min(100.0, value))

        if self._wing is not None:
            try:
                self._wing.send_float(f"/ch/{ch}/pan", value)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} pan: {exc}")
                return {"success": False, "error": str(exc)}

        self._channel_state.setdefault(ch, {})["pan"] = value
        logger.info(f"Ch {ch}: pan -> {value:.0f}")
        return {"success": True, "ch": ch, "pan": value}

    async def set_channel_mute(self, ch: int, muted: bool) -> Dict[str, Any]:
        """
        Mute or unmute a channel.

        Args:
            ch: Channel number (1-40).
            muted: True to mute, False to unmute.

        Returns:
            Result dict.
        """
        mute_val = 1 if muted else 0

        if self._wing is not None:
            try:
                self._wing.send_int(f"/ch/{ch}/mute", mute_val)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} mute: {exc}")
                return {"success": False, "error": str(exc)}

        self._channel_state.setdefault(ch, {})["mute"] = muted
        logger.info(f"Ch {ch}: mute -> {'ON' if muted else 'OFF'}")
        return {"success": True, "ch": ch, "muted": muted}

    async def set_channel_hpf(self, ch: int, frequency: float) -> Dict[str, Any]:
        """
        Set high-pass filter frequency on a channel.

        Args:
            ch: Channel number (1-40).
            frequency: HPF frequency in Hz (20-2000).

        Returns:
            Result dict.
        """
        frequency = max(20.0, min(2000.0, frequency))

        if self._wing is not None:
            try:
                # Enable low-cut filter and set frequency
                self._wing.send_int(f"/ch/{ch}/flt/lc", 1)
                self._wing.send_float(f"/ch/{ch}/flt/lcf", frequency)
            except Exception as exc:
                logger.error(f"OSC send failed for ch {ch} HPF: {exc}")
                return {"success": False, "error": str(exc)}

        self._channel_state.setdefault(ch, {})["hpf"] = frequency
        logger.info(f"Ch {ch}: HPF -> {frequency:.0f} Hz")
        return {"success": True, "ch": ch, "hpf_frequency": frequency}

    async def get_channel_levels(self, ch: int) -> Dict[str, Any]:
        """
        Get current levels for a channel.

        Args:
            ch: Channel number (1-40).

        Returns:
            Dict with level information (fader, meter readings, etc.).
        """
        levels: Dict[str, Any] = {"ch": ch}

        # Return cached state
        cached = self._channel_state.get(ch, {})
        levels["fader"] = cached.get("fader", 0.0)
        levels["pan"] = cached.get("pan", 0.0)
        levels["mute"] = cached.get("mute", False)
        levels["hpf"] = cached.get("hpf", 0.0)
        levels["instrument"] = self._channel_instruments.get(ch, "unknown")

        # If wing client available, query live meter data
        if self._wing is not None:
            try:
                # Query current state from mixer
                state = getattr(self._wing, "state", {})
                fdr_addr = f"/ch/{ch}/fdr"
                if fdr_addr in state:
                    levels["fader_live"] = state[fdr_addr]
            except Exception as exc:
                logger.debug(f"Could not query live levels for ch {ch}: {exc}")

        return levels

    async def get_mix_quality(self) -> Dict[str, Any]:
        """
        Get overall mix quality metrics.

        Evaluates the current mix state and returns quality indicators.

        Returns:
            Dict with quality metrics: headroom, channel_count, issues list.
        """
        metrics: Dict[str, Any] = {
            "active_channels": len(self._channel_state),
            "classified_channels": len(self._channel_instruments),
            "issues": [],
        }

        # Check for common issues
        for ch, state in self._channel_state.items():
            fader = state.get("fader", 0.0)
            instrument = self._channel_instruments.get(ch, "unknown")

            # Check for channels pushed too hot
            if fader > 5.0:
                metrics["issues"].append(
                    f"Ch {ch} ({instrument}): fader at {fader:.1f} dB, "
                    f"very hot — consider pulling back"
                )

            # Check for muted channels that should be active
            if state.get("mute") and instrument != "unknown":
                metrics["issues"].append(
                    f"Ch {ch} ({instrument}): classified but muted"
                )

            # Check for missing HPF
            if state.get("hpf", 0.0) < 20.1 and instrument not in (
                "kick", "bass", "sub", "playback", "djTrack", "unknown"
            ):
                metrics["issues"].append(
                    f"Ch {ch} ({instrument}): no HPF set — consider enabling"
                )

        # Overall headroom estimate
        max_fader = max(
            (s.get("fader", -144.0) for s in self._channel_state.values()),
            default=-144.0,
        )
        metrics["max_fader_db"] = max_fader
        metrics["estimated_headroom_db"] = MAX_FADER_DB - max_fader

        # Score: 0-100 based on number of issues
        issue_count = len(metrics["issues"])
        metrics["score"] = max(0, 100 - issue_count * 10)

        return metrics

    async def search_knowledge(self, query: str) -> Dict[str, Any]:
        """
        Search the mixing knowledge base.

        Args:
            query: Natural language search query.

        Returns:
            Dict with search results.
        """
        results = self.knowledge_base.search(query, top_k=5)
        return {
            "query": query,
            "results": [
                {"doc_id": doc_id, "text": text[:500], "score": round(score, 3)}
                for doc_id, text, score in results
            ],
            "count": len(results),
        }

    async def apply_preset(self, ch: int, preset_name: str) -> Dict[str, Any]:
        """
        Apply a full instrument preset to a channel.

        This applies HPF, EQ, compressor, gate, and pan settings
        based on the rule engine's preset for the given instrument type.

        Args:
            ch: Channel number (1-40).
            preset_name: Instrument type / preset name (e.g. 'kick', 'leadVocal').

        Returns:
            Dict with all applied settings.
        """
        preset = self.rule_engine.get_full_channel_preset(preset_name)
        results: Dict[str, Any] = {"ch": ch, "preset": preset_name, "applied": {}}

        # HPF
        hpf_result = await self.set_channel_hpf(ch, preset["hpf"])
        results["applied"]["hpf"] = hpf_result

        # EQ bands
        eq_results = []
        for i, band in enumerate(preset["eq"]):
            band_num = i + 1  # Wing uses 1-4 for parametric bands
            eq_result = await self.set_channel_eq(
                ch, band_num, band["freq"], band["gain"], band["q"],
            )
            eq_results.append(eq_result)
        results["applied"]["eq"] = eq_results

        # Enable EQ
        if self._wing is not None:
            try:
                self._wing.send_int(f"/ch/{ch}/eq/on", 1)
            except Exception:
                pass

        # Compressor
        comp = preset["compressor"]
        comp_result = await self.set_channel_compressor(
            ch, comp["threshold"], comp["ratio"],
            comp["attack"], comp["release"],
        )
        results["applied"]["compressor"] = comp_result

        # Gate (only if recommended for this instrument)
        if self.rule_engine.should_enable_gate(preset_name):
            gate = preset["gate"]
            gate_result = await self.set_channel_gate(
                ch, gate["threshold"], gate["attack"],
                gate["hold"], gate["release"],
            )
            results["applied"]["gate"] = gate_result
        else:
            # Disable gate for instruments that don't need it
            if self._wing is not None:
                try:
                    self._wing.send_int(f"/ch/{ch}/gate/on", 0)
                except Exception:
                    pass
            results["applied"]["gate"] = {"enabled": False}

        # Pan
        pan_result = await self.set_channel_pan(ch, preset["pan"])
        results["applied"]["pan"] = pan_result

        # Track instrument assignment
        self._channel_instruments[ch] = preset_name

        logger.info(f"Ch {ch}: applied full '{preset_name}' preset")
        results["success"] = True
        return results

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(
        self, tool_name: str, parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a named tool with the given parameters.

        Args:
            tool_name: One of the 11 registered tool names.
            parameters: Dict of parameter name -> value.

        Returns:
            Tool result dict.
        """
        tool_fn = self._tools.get(tool_name)
        if tool_fn is None:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            result = await tool_fn(**parameters)
            return result
        except TypeError as exc:
            return {"error": f"Invalid parameters for {tool_name}: {exc}"}
        except Exception as exc:
            logger.error(f"Tool {tool_name} failed: {exc}", exc_info=True)
            return {"error": f"Tool execution failed: {exc}"}

    # ------------------------------------------------------------------
    # 3-tier routing
    # ------------------------------------------------------------------

    async def process_command(self, command: str) -> Dict[str, Any]:
        """
        Process a natural language mixing command using 3-tier routing.

        Tier 1: Check if the command maps to a deterministic rule.
        Tier 2: Use local LLM for complex reasoning.
        Tier 3: Fall back to cloud LLM.

        Args:
            command: Natural language command string.

        Returns:
            Dict with response and any actions taken.
        """
        command_lower = command.lower().strip()
        response: Dict[str, Any] = {"command": command, "tier": 0, "actions": []}

        # ----- Tier 1: Rule-based pattern matching -----
        tier1_result = self._try_rule_based(command_lower)
        if tier1_result is not None:
            response["tier"] = 1
            response["tier_name"] = "rule_engine"
            response["actions"] = tier1_result.get("actions", [])
            response["response"] = tier1_result.get("response", "")
            # Execute any actions
            for action in response["actions"]:
                result = await self.execute_tool(
                    action["tool"], action["parameters"],
                )
                action["result"] = result
            return response

        # ----- Tier 2/3: LLM with knowledge augmentation -----
        # Augment the prompt with relevant knowledge
        knowledge_results = self.knowledge_base.search(command, top_k=3)
        context_parts = []
        for doc_id, text, score in knowledge_results:
            if score > 0.1:
                context_parts.append(text[:300])

        augmented_prompt = self._build_llm_prompt(command, context_parts)

        try:
            llm_response = await self._llm_chain.chat(augmented_prompt)
            tier_name = self._llm_chain.last_successful_client or "unknown"
            response["tier"] = 2 if "Ollama" in tier_name else 3
            response["tier_name"] = tier_name
            response["response"] = llm_response

            # Try to parse tool calls from LLM response
            parsed_actions = self._parse_llm_actions(llm_response)
            if parsed_actions:
                response["actions"] = parsed_actions
                for action in response["actions"]:
                    result = await self.execute_tool(
                        action["tool"], action["parameters"],
                    )
                    action["result"] = result

        except RuntimeError as exc:
            logger.warning(f"All LLM tiers failed: {exc}")
            response["tier"] = 0
            response["tier_name"] = "fallback"
            response["response"] = (
                "I could not process this command through any available AI tier. "
                "The rule engine did not match, and LLM services are unavailable. "
                "Try a more specific command like 'apply kick preset to channel 1'."
            )

        return response

    def _try_rule_based(self, command: str) -> Optional[Dict[str, Any]]:
        """
        Try to handle the command with deterministic rules.

        Returns None if the command doesn't match any rule patterns.
        """
        # Pattern: "apply {preset} to channel {n}" or "set ch {n} as {preset}"
        m = re.match(
            r'(?:apply|set)\s+(?:(?:ch(?:annel)?\s*(\d+))\s+(?:as|to)\s+(\w+)'
            r'|(\w+)\s+(?:preset\s+)?(?:to|on)\s+ch(?:annel)?\s*(\d+))',
            command,
        )
        if m:
            if m.group(1) and m.group(2):
                ch, preset = int(m.group(1)), m.group(2)
            else:
                preset, ch = m.group(3), int(m.group(4))
            return {
                "response": f"Applying {preset} preset to channel {ch}",
                "actions": [
                    {"tool": "apply_preset", "parameters": {"ch": ch, "preset_name": preset}},
                ],
            }

        # Pattern: "mute channel {n}" / "unmute channel {n}"
        m = re.match(r'(un)?mute\s+ch(?:annel)?\s*(\d+)', command)
        if m:
            muted = m.group(1) is None  # "mute" = True, "unmute" = False
            ch = int(m.group(2))
            return {
                "response": f"{'Muting' if muted else 'Unmuting'} channel {ch}",
                "actions": [
                    {"tool": "set_channel_mute", "parameters": {"ch": ch, "muted": muted}},
                ],
            }

        # Pattern: "set channel {n} fader to {value}"
        m = re.match(
            r'set\s+ch(?:annel)?\s*(\d+)\s+fader\s+(?:to\s+)?(-?\d+(?:\.\d+)?)',
            command,
        )
        if m:
            ch = int(m.group(1))
            value = float(m.group(2))
            return {
                "response": f"Setting channel {ch} fader to {value:.1f} dB",
                "actions": [
                    {"tool": "set_channel_fader", "parameters": {"ch": ch, "value": value}},
                ],
            }

        # Pattern: "set channel {n} pan to {value}"
        m = re.match(
            r'set\s+ch(?:annel)?\s*(\d+)\s+pan\s+(?:to\s+)?(-?\d+(?:\.\d+)?)',
            command,
        )
        if m:
            ch = int(m.group(1))
            value = float(m.group(2))
            return {
                "response": f"Setting channel {ch} pan to {value:.0f}",
                "actions": [
                    {"tool": "set_channel_pan", "parameters": {"ch": ch, "value": value}},
                ],
            }

        # Pattern: "set channel {n} hpf to {freq}"
        m = re.match(
            r'set\s+ch(?:annel)?\s*(\d+)\s+(?:hpf|high[\s-]?pass)\s+(?:to\s+)?(\d+(?:\.\d+)?)',
            command,
        )
        if m:
            ch = int(m.group(1))
            freq = float(m.group(2))
            return {
                "response": f"Setting channel {ch} HPF to {freq:.0f} Hz",
                "actions": [
                    {"tool": "set_channel_hpf", "parameters": {"ch": ch, "frequency": freq}},
                ],
            }

        # Pattern: "feedback at {freq} hz"
        m = re.match(r'feedback\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*(?:hz)?', command)
        if m:
            freq = float(m.group(1))
            notch = self.rule_engine.handle_feedback(freq)
            return {
                "response": (
                    f"Feedback detected at {freq:.0f} Hz. "
                    f"Applying notch: {notch['gain']:.0f} dB cut, "
                    f"Q={notch['q']:.1f} (severity: {notch['severity']})"
                ),
                "actions": [],  # Feedback handling returns advice; agent decides which channel
            }

        # Pattern: "soundcheck" / "run soundcheck"
        if "soundcheck" in command:
            return {
                "response": "Starting automated soundcheck...",
                "actions": [{"tool": "_run_soundcheck", "parameters": {}}],
            }

        return None

    def _build_llm_prompt(
        self, command: str, context_parts: List[str],
    ) -> str:
        """Build an augmented prompt for the LLM with knowledge context."""
        parts = [
            "You are an AI mixing engineer assistant for a Behringer Wing Rack "
            "digital mixer. You have access to the following tools:\n",
        ]

        for tool in TOOL_DEFINITIONS:
            params_str = ", ".join(
                f"{k}: {v}" for k, v in tool["parameters"].items()
            )
            parts.append(f"- {tool['name']}({params_str}): {tool['description']}")

        parts.append("\n")

        if context_parts:
            parts.append("Relevant knowledge from the mixing database:")
            for ctx in context_parts:
                parts.append(f"  {ctx}")
            parts.append("")

        # Current mixer state summary
        if self._channel_instruments:
            parts.append("Current channel assignments:")
            for ch, inst in sorted(self._channel_instruments.items()):
                fader = self._channel_state.get(ch, {}).get("fader", "?")
                parts.append(f"  Ch {ch}: {inst} (fader: {fader})")
            parts.append("")

        parts.append(f"User command: {command}")
        parts.append(
            "\nRespond with a brief explanation followed by any tool calls "
            "in JSON format: {\"tool\": \"name\", \"parameters\": {...}}"
        )

        return "\n".join(parts)

    def _parse_llm_actions(
        self, response: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse tool call JSON objects from LLM response text.

        Looks for JSON objects with 'tool' and 'parameters' keys.
        """
        actions: List[Dict[str, Any]] = []

        # Find all JSON-like objects in the response
        json_pattern = re.compile(r'\{[^{}]*"tool"\s*:\s*"[^"]+?"[^{}]*\}')
        matches = json_pattern.findall(response)

        for match in matches:
            try:
                obj = json.loads(match)
                if "tool" in obj and "parameters" in obj:
                    tool_name = obj["tool"]
                    if tool_name in self._tools:
                        actions.append({
                            "tool": tool_name,
                            "parameters": obj["parameters"],
                        })
                    else:
                        logger.debug(f"LLM suggested unknown tool: {tool_name}")
            except json.JSONDecodeError:
                continue

        return actions

    # ------------------------------------------------------------------
    # Soundcheck
    # ------------------------------------------------------------------

    async def run_soundcheck(
        self,
        channels: Optional[List[int]] = None,
        channel_names: Optional[Dict[int, str]] = None,
        classifier_fn: Optional[Callable[[str], Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run an automated soundcheck across channels.

        For each channel:
        1. Read the channel name from the mixer (or use provided names).
        2. Classify the instrument type.
        3. Apply the appropriate rule-engine preset.
        4. Log the applied settings.

        Args:
            channels: List of channel numbers to check (default: 1-40).
            channel_names: Optional mapping of channel number -> name.
            classifier_fn: Optional function(name) -> instrument_type.
                           Defaults to keyword matching from channel_recognizer.

        Returns:
            Dict with per-channel results and overall summary.
        """
        if channels is None:
            channels = list(range(1, 41))

        # Try to import the project's channel recognizer
        if classifier_fn is None:
            classifier_fn = self._default_classifier

        results: Dict[str, Any] = {
            "channels": {},
            "classified": 0,
            "unclassified": 0,
            "errors": [],
        }

        for ch in channels:
            ch_name = ""
            if channel_names and ch in channel_names:
                ch_name = channel_names[ch]
            elif self._wing is not None:
                # Try to get name from mixer state
                try:
                    state = getattr(self._wing, "state", {})
                    name_addr = f"/ch/{ch}/name"
                    ch_name = state.get(name_addr, "")
                except Exception:
                    pass

            if not ch_name:
                results["channels"][ch] = {"status": "skipped", "reason": "no name"}
                results["unclassified"] += 1
                continue

            # Classify instrument
            instrument = classifier_fn(ch_name)
            if instrument is None:
                results["channels"][ch] = {
                    "status": "unclassified",
                    "name": ch_name,
                }
                results["unclassified"] += 1
                continue

            # Apply preset
            try:
                preset_result = await self.apply_preset(ch, instrument)
                results["channels"][ch] = {
                    "status": "configured",
                    "name": ch_name,
                    "instrument": instrument,
                    "preset": preset_result,
                }
                results["classified"] += 1

                # Small delay between channels to avoid OSC flooding
                await asyncio.sleep(SOUNDCHECK_SETTLE_TIME)

            except Exception as exc:
                logger.error(f"Soundcheck failed for ch {ch}: {exc}")
                results["channels"][ch] = {
                    "status": "error",
                    "name": ch_name,
                    "instrument": instrument,
                    "error": str(exc),
                }
                results["errors"].append(f"Ch {ch} ({ch_name}): {exc}")

        total = len(channels)
        results["summary"] = (
            f"Soundcheck complete: {results['classified']}/{total} channels "
            f"classified and configured, {results['unclassified']} unclassified, "
            f"{len(results['errors'])} errors"
        )
        logger.info(results["summary"])
        return results

    @staticmethod
    def _default_classifier(channel_name: str) -> Optional[str]:
        """
        Default channel name classifier using keyword matching.

        Tries to import channel_recognizer from the project; falls back
        to simple keyword matching.
        """
        try:
            import sys
            # Add backend to path if needed
            backend_dir = str(Path(__file__).parent.parent)
            if backend_dir not in sys.path:
                sys.path.insert(0, backend_dir)
            from channel_recognizer import recognize_instrument
            return recognize_instrument(channel_name)
        except ImportError:
            pass

        # Fallback: simple keyword matching
        name_lower = channel_name.lower()
        keyword_map = {
            "kick": "kick", "bd": "kick",
            "snare": "snare", "sd": "snare", "sn": "snare",
            "tom": "tom",
            "hihat": "hihat", "hh": "hihat", "hi-hat": "hihat",
            "overhead": "overheads", "oh": "overheads",
            "room": "room",
            "bass": "bass",
            "gtr": "electricGuitar", "guitar": "electricGuitar",
            "acoustic": "acousticGuitar", "agtr": "acousticGuitar",
            "vox": "leadVocal", "vocal": "leadVocal", "voice": "leadVocal",
            "bvox": "backVocal", "bgv": "backVocal", "choir": "backVocal",
            "synth": "synth", "keys": "synth", "keyboard": "synth",
            "piano": "piano",
            "accordion": "accordion",
            "trumpet": "trumpet",
            "sax": "saxophone",
            "playback": "playback", "track": "playback", "pb": "playback",
        }
        for keyword, instrument in keyword_map.items():
            if keyword in name_lower:
                return instrument
        return None

    # ------------------------------------------------------------------
    # Feedback emergency handler
    # ------------------------------------------------------------------

    async def handle_feedback_emergency(
        self, ch: int, frequency_hz: float,
    ) -> Dict[str, Any]:
        """
        Emergency feedback suppression — applies immediate notch EQ.

        This bypasses the normal gain-change safety limits because
        feedback must be stopped immediately.

        Args:
            ch: Channel where feedback is detected.
            frequency_hz: Detected feedback frequency in Hz.

        Returns:
            Dict with actions taken.
        """
        notch = self.rule_engine.handle_feedback(frequency_hz)

        # Find an available EQ band (prefer band 4 as "utility" band)
        result = await self.set_channel_eq(
            ch=ch,
            band=4,
            freq=notch["freq"],
            gain=notch["gain"],
            q=notch["q"],
        )

        logger.warning(
            f"FEEDBACK EMERGENCY: Ch {ch} @ {frequency_hz:.0f} Hz — "
            f"applied {notch['gain']:.0f} dB notch, Q={notch['q']:.1f}"
        )

        return {
            "action": "feedback_suppression",
            "ch": ch,
            "frequency": frequency_hz,
            "notch": notch,
            "eq_result": result,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Shut down LLM clients and release resources."""
        await self._llm_chain.close()
        logger.info("AIAgent shut down")
