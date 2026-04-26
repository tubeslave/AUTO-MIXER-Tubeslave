#!/usr/bin/env python3
"""Offline multitrack mix using the same rule/agent path as live control.

The script treats each WAV file as a console channel, runs a virtual mixer
through MixingAgent + RuleEngine, applies channel strip DSP, then exports
a mastered MP3 for listening checks.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import yaml
from scipy.signal import butter, filtfilt, lfilter

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from ai.agent import AgentAction, AgentMode, MixingAgent  # noqa: E402
from ai.knowledge_base import KnowledgeBase  # noqa: E402
from ai.llm_client import LLMClient  # noqa: E402
from ai.rule_engine import RuleEngine  # noqa: E402
from auto_mastering import AutoMaster, MasteringResult  # noqa: E402
from auto_phase_gcc_phat import GCCPHATAnalyzer  # noqa: E402
from autofoh_analysis import build_fractional_octave_bands, build_stem_contribution_matrix, extract_analysis_features  # noqa: E402
from autofoh_detectors import (  # noqa: E402
    HarshnessExcessDetector,
    LeadMaskingAnalyzer,
    LowEndAnalyzer,
    MudExcessDetector,
    SibilanceExcessDetector,
    aggregate_stem_features,
)
from autofoh_models import ConfidenceRisk, DetectedProblem, RuntimeState  # noqa: E402
from autofoh_safety import (  # noqa: E402
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    ChannelGainMove,
    ChannelEQMove,
    ChannelFaderMove,
    TypedCorrectionAction,
)
from auto_fx import AutoFXPlanner, FXBusDecision, FXPlan, FXSendDecision  # noqa: E402
from channel_recognizer import classification_from_legacy_preset  # noqa: E402
from cross_adaptive_eq import CrossAdaptiveEQ  # noqa: E402
from ml.style_transfer import InstrumentStyle, StyleProfile, StyleTransfer  # noqa: E402
from perceptual import PerceptualConfig, PerceptualEvaluator  # noqa: E402
from source_knowledge import (  # noqa: E402
    DecisionTrace,
    FeedbackRecord,
    RuleMatch,
    SourceGroundedConfig,
    SourceKnowledgeLayer,
)


DRUM_INSTRUMENTS = {
    "kick",
    "snare",
    "rack_tom",
    "floor_tom",
    "hi_hat",
    "ride",
    "percussion",
}

REFERENCE_STYLE_INSTRUMENTS = {
    "lead_vocal": "vocals",
    "backing_vocal": "vocals",
    "kick": "kick",
    "snare": "snare",
    "rack_tom": "toms",
    "floor_tom": "toms",
    "hi_hat": "hihat",
    "ride": "hihat",
    "overhead": "overheads",
    "oh_l": "overheads",
    "oh_r": "overheads",
    "cymbals": "overheads",
    "bass": "bass",
    "bass_di": "bass",
    "bass_mic": "bass",
    "synth_bass": "bass",
    "guitar": "electric_guitar",
    "electric_guitar": "electric_guitar",
    "lead_guitar": "electric_guitar",
    "rhythm_guitar": "electric_guitar",
    "acoustic_guitar": "acoustic_guitar",
    "keys": "keys",
    "piano": "keys",
    "organ": "keys",
    "synth": "keys",
    "pad": "keys",
    "lead_synth": "keys",
    "percussion": "percussion",
}

REFERENCE_COMPRESSOR_SKIP = {
    "hi_hat",
    "ride",
    "overhead",
    "oh_l",
    "oh_r",
    "cymbals",
    "playback",
    "tracks",
    "music",
    "fx_return",
    "reverb_return",
    "delay_return",
    "room",
}

REFERENCE_AUDIO_SUFFIXES = {
    ".wav",
    ".wave",
    ".aif",
    ".aiff",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
}

REFERENCE_PRESET_SUFFIXES = {
    ".json",
}

BASS_INSTRUMENTS = {
    "bass",
    "bass_guitar",
    "bass_di",
    "bass_mic",
    "synth_bass",
}

CYMBAL_INSTRUMENTS = {
    "hi_hat",
    "ride",
    "overhead",
    "oh_l",
    "oh_r",
    "cymbals",
}

WINDOW_SPACE_COMPETITOR_INSTRUMENTS = {
    "accordion",
    "backing_vocal",
    "guitar",
    "electric_guitar",
    "lead_guitar",
    "rhythm_guitar",
    "acoustic_guitar",
    "keys",
    "piano",
    "organ",
    "synth",
    "pad",
    "lead_synth",
    "playback",
}

FREQUENCY_WINDOW_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "low_end_foundation",
        "label": "Low End Foundation",
        "low_hz": 20.0,
        "high_hz": 120.0,
        "focus": "kick, bass, sub stability, mono foundation",
        "action_mode": "report_only",
    },
    {
        "id": "warmth_mud",
        "label": "Warmth / Mud",
        "low_hz": 120.0,
        "high_hz": 500.0,
        "focus": "body, mud, room build-up, low-mid masking",
        "action_mode": "cleanup_music",
        "center_hz": 320.0,
        "q": 1.15,
    },
    {
        "id": "core_mids",
        "label": "Core Mids",
        "low_hz": 500.0,
        "high_hz": 1000.0,
        "focus": "musical skeleton, note readability, center integrity",
        "action_mode": "report_only",
    },
    {
        "id": "vocal_conflict",
        "label": "Vocal Conflict",
        "low_hz": 700.0,
        "high_hz": 1500.0,
        "focus": "lead vocal against guitars, keys, playback, backing stack",
        "action_mode": "clear_vocal_space",
        "center_hz": 1100.0,
        "q": 1.3,
    },
    {
        "id": "presence_harshness",
        "label": "Presence / Harshness",
        "low_hz": 1500.0,
        "high_hz": 6000.0,
        "focus": "presence, attack, intelligibility, harshness buildup",
        "action_mode": "clear_vocal_space",
        "center_hz": 2900.0,
        "q": 1.45,
    },
    {
        "id": "air_sibilance",
        "label": "Air / Sibilance",
        "low_hz": 6000.0,
        "high_hz": 16000.0,
        "focus": "hats, cymbals, sibilance, air integration",
        "action_mode": "cymbal_control",
        "center_hz": 8500.0,
        "q": 1.2,
    },
)

BAND_ANALYSIS_WINDOW_SEC = 18.0
FREQUENCY_WINDOW_PREVIEW_SEC = 8.0
ANALYZER_RENDER_PREVIEW_SEC = 24.0
DEFAULT_AUTOFOH_ANALYZER_ROUNDS = 1
DEFAULT_RENDER_CACHE_MAX_MB = 1536.0
FAST_COMPRESSOR_MIN_SAMPLES = 8192
FAST_COMPRESSOR_FRAME_MS = 2.0
MIRROR_EQ_CENTERS_HZ = (63.0, 125.0, 250.0, 350.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)

GENRE_REFERENCE_ALIASES = {
    "edm": "electronic",
    "dance": "electronic",
    "dance_pop": "electronic",
    "house": "electronic",
    "club": "electronic",
    "synthpop": "synth_pop",
    "synth_pop": "synth_pop",
    "electropop": "synth_pop",
    "alt_pop": "pop",
    "hiphop": "hip_hop",
    "hip_hop": "hip_hop",
    "rap": "hip_hop",
    "trap": "hip_hop",
    "trap_pop": "hip_hop",
    "rnb": "r_and_b",
    "r_b": "r_and_b",
    "r_and_b": "r_and_b",
    "neo_soul": "r_and_b",
    "pop_rock": "rock",
    "alt_rock": "rock",
    "alternative": "rock",
    "indie_rock": "rock",
}

GENRE_REFERENCE_STARTS: dict[str, dict[str, Any]] = {
    "pop": {
        "label": "Modern Pop",
        "notes": [
            "Internet-curated seed built from modern reference-track recommendations by iZotope and Mastering The Mix.",
            "Use this as a starting balance target when no local audio reference is supplied, or as a prior before a song-specific reference overrides it.",
        ],
        "references": [
            {
                "song": "Anti-Hero",
                "artist": "Taylor Swift",
                "mix_engineer": "Serban Ghenea",
                "source_url": "https://www.izotope.com/en/learn/mixing-reference-tracks.html",
                "why": "Current chart-pop vocal depth, synth-pop width, and modern top-end gloss.",
            },
            {
                "song": "New Rules",
                "artist": "Dua Lipa",
                "source_url": "https://www.masteringthemix.com/blogs/learn/best-reference-track-for-all-genres",
                "why": "Huge but controlled bass with wide midrange musical elements.",
            },
            {
                "song": "Get Lucky",
                "artist": "Daft Punk feat. Nile Rodgers and Pharrell Williams",
                "mix_engineer": "Mick Guzauski",
                "source_url": "https://www.izotope.com/en/learn/4-popular-mixing-reference-tracks-and-why-they-work",
                "why": "Low-end definition, depth of field, and wide-but-stable imaging.",
            },
        ],
        "tilt": {
            "weight_60_120_vs_plateau_db": 0.2,
            "sub_35_60_vs_plateau_db": -2.8,
            "high_4500_12000_vs_plateau_db": -0.2,
            "plateau_spread_db": 5.4,
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": 5.8,
            "lead_over_bgv_presence_db": 5.4,
            "kick_over_bass_55_95_db": 0.8,
        },
        "kick": {
            "click_share_in_master": 0.03,
            "click_share_in_drums": 0.1,
            "dynamic_range_max_db": 16.8,
            "click_minus_punch_db": -17.8,
            "box_minus_click_db": -0.4,
        },
        "style_summary": {
            "dynamic_range_db": 7.2,
            "stereo_width": 0.46,
            "crest_factor_db": 11.4,
        },
    },
    "synth_pop": {
        "label": "Synth Pop / Alt Pop",
        "notes": [
            "Internet-curated seed built around modern synth-pop references with glossy vocals, stable low end, and a flatter compensated plateau.",
        ],
        "references": [
            {
                "song": "Anti-Hero",
                "artist": "Taylor Swift",
                "mix_engineer": "Serban Ghenea",
                "source_url": "https://www.izotope.com/en/learn/mixing-reference-tracks.html",
                "why": "Modern synth/vocal/percussion balance with front vocal and controlled width.",
            },
            {
                "song": "Get Lucky",
                "artist": "Daft Punk feat. Nile Rodgers and Pharrell Williams",
                "mix_engineer": "Mick Guzauski",
                "source_url": "https://www.izotope.com/en/learn/4-popular-mixing-reference-tracks-and-why-they-work",
                "why": "Wide and deep imaging with excellent low-end definition.",
            },
            {
                "song": "demon time (with BAYLI)",
                "artist": "Mura Masa",
                "mix_engineer": "Nathan Boddy",
                "source_url": "https://www.izotope.com/en/learn/mixing-reference-tracks.html",
                "why": "Active stereo motion and contemporary electronic textures around the vocal.",
            },
        ],
        "tilt": {
            "weight_60_120_vs_plateau_db": 0.0,
            "sub_35_60_vs_plateau_db": -2.5,
            "high_4500_12000_vs_plateau_db": 0.0,
            "plateau_spread_db": 5.1,
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": 5.4,
            "lead_over_bgv_presence_db": 5.2,
            "kick_over_bass_55_95_db": 0.7,
        },
        "kick": {
            "click_share_in_master": 0.028,
            "click_share_in_drums": 0.092,
            "dynamic_range_max_db": 17.0,
            "click_minus_punch_db": -18.0,
            "box_minus_click_db": -0.3,
        },
        "style_summary": {
            "dynamic_range_db": 6.6,
            "stereo_width": 0.5,
            "crest_factor_db": 11.0,
        },
    },
    "electronic": {
        "label": "Electronic / EDM",
        "notes": [
            "Internet-curated seed built from modern electronic reference picks that stress kick/sub trading, sidechain groove, and wide but uncluttered tops.",
        ],
        "references": [
            {
                "song": "Riptide",
                "artist": "The Chainsmokers",
                "source_url": "https://www.izotope.com/en/learn/10-great-reference-mixes-for-electronic-music",
                "why": "Mainstream electronic low-end trading between kick and synth bass across sections.",
            },
            {
                "song": "Spacetrippy",
                "artist": "Black Hertz",
                "source_url": "https://www.izotope.com/en/learn/10-great-reference-mixes-for-electronic-music",
                "why": "Clear sidechain groove, deep bass, and sparkling but controlled highs.",
            },
            {
                "song": "Get Lucky",
                "artist": "Daft Punk feat. Nile Rodgers and Pharrell Williams",
                "mix_engineer": "Mick Guzauski",
                "source_url": "https://www.izotope.com/en/learn/4-popular-mixing-reference-tracks-and-why-they-work",
                "why": "Benchmark for frequency extension, stereo imaging, and depth of field.",
            },
        ],
        "tilt": {
            "weight_60_120_vs_plateau_db": 1.2,
            "sub_35_60_vs_plateau_db": -1.0,
            "high_4500_12000_vs_plateau_db": -0.3,
            "plateau_spread_db": 5.8,
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": 4.6,
            "lead_over_bgv_presence_db": 4.4,
            "kick_over_bass_55_95_db": 1.3,
        },
        "kick": {
            "click_share_in_master": 0.04,
            "click_share_in_drums": 0.13,
            "dynamic_range_max_db": 15.8,
            "click_minus_punch_db": -16.2,
            "box_minus_click_db": -0.8,
        },
        "style_summary": {
            "dynamic_range_db": 6.0,
            "stereo_width": 0.58,
            "crest_factor_db": 10.8,
        },
    },
    "hip_hop": {
        "label": "Hip-Hop / Rap",
        "notes": [
            "Internet-curated seed built from hip-hop references that prioritize mono-safe low end, vocal pocket clarity, and modern but not washy ambience.",
        ],
        "references": [
            {
                "song": "rockstar",
                "artist": "Post Malone feat. 21 Savage",
                "mix_engineer": "Manny Marroquin",
                "source_url": "https://www.izotope.com/en/learn/4-popular-mixing-reference-tracks-and-why-they-work",
                "why": "Great space and dynamics with a modern crossover vocal picture.",
            },
            {
                "song": "KOD",
                "artist": "J. Cole",
                "mix_engineer": "Juro Davis",
                "source_url": "https://www.izotope.com/en/learn/8-killer-reference-mixes-for-hip-hop.html",
                "why": "Clean, modern, heavy low end with clear centered lead vocal.",
            },
            {
                "song": "King's Dead",
                "artist": "Jay Rock, Kendrick Lamar, Future, James Blake",
                "mix_engineer": "Matt Schaeffer",
                "source_url": "https://www.izotope.com/en/learn/8-killer-reference-mixes-for-hip-hop.html",
                "why": "Complex kick patterns, aggressive bass, and managed upper-mid harshness.",
            },
        ],
        "tilt": {
            "weight_60_120_vs_plateau_db": 1.7,
            "sub_35_60_vs_plateau_db": -0.4,
            "high_4500_12000_vs_plateau_db": -1.2,
            "plateau_spread_db": 5.6,
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": 5.0,
            "lead_over_bgv_presence_db": 4.8,
            "kick_over_bass_55_95_db": 0.9,
        },
        "kick": {
            "click_share_in_master": 0.026,
            "click_share_in_drums": 0.088,
            "dynamic_range_max_db": 16.0,
            "click_minus_punch_db": -18.5,
            "box_minus_click_db": -0.2,
        },
        "style_summary": {
            "dynamic_range_db": 6.4,
            "stereo_width": 0.34,
            "crest_factor_db": 11.2,
        },
    },
    "r_and_b": {
        "label": "Modern R&B",
        "notes": [
            "Internet-curated seed built from contemporary crossover R&B references with roomy vocal production, softer low-end ownership, and smoother top-end density.",
        ],
        "references": [
            {
                "song": "Wild Thoughts",
                "artist": "DJ Khaled feat. Rihanna and Bryson Tiller",
                "source_url": "https://www.masteringthemix.com/blogs/learn/best-reference-track-for-all-genres",
                "why": "Consistent full-spectrum tonal balance and polished vocal production.",
            },
            {
                "song": "All The Stars",
                "artist": "Kendrick Lamar and SZA",
                "source_url": "https://www.masteringthemix.com/blogs/learn/best-reference-track-for-all-genres",
                "why": "Detailed vocals without harshness and an even full-range balance.",
            },
            {
                "song": "Anti-Hero",
                "artist": "Taylor Swift",
                "mix_engineer": "Serban Ghenea",
                "source_url": "https://www.izotope.com/en/learn/mixing-reference-tracks.html",
                "why": "Useful modern vocal-depth anchor when the R&B production leans pop.",
            },
        ],
        "tilt": {
            "weight_60_120_vs_plateau_db": 0.5,
            "sub_35_60_vs_plateau_db": -1.8,
            "high_4500_12000_vs_plateau_db": -0.6,
            "plateau_spread_db": 5.0,
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": 5.4,
            "lead_over_bgv_presence_db": 5.1,
            "kick_over_bass_55_95_db": 0.6,
        },
        "kick": {
            "click_share_in_master": 0.024,
            "click_share_in_drums": 0.082,
            "dynamic_range_max_db": 17.2,
            "click_minus_punch_db": -19.2,
            "box_minus_click_db": 0.0,
        },
        "style_summary": {
            "dynamic_range_db": 7.4,
            "stereo_width": 0.42,
            "crest_factor_db": 11.8,
        },
    },
    "rock": {
        "label": "Modern Rock",
        "notes": [
            "Internet-curated seed built from modern rock reference-track roundups and pro mixer picks.",
        ],
        "references": [
            {
                "song": "Chaise Longue",
                "artist": "Wet Leg",
                "mix_engineer": "Alan Moulder",
                "source_url": "https://www.izotope.com/en/learn/mixing-reference-tracks.html",
                "why": "Heavy snare, intimate vocal, and controlled non-hyped top end.",
            },
            {
                "song": "Jumpsuit",
                "artist": "Twenty One Pilots",
                "source_url": "https://www.izotope.com/en/learn/pro-reference-tracks.html",
                "why": "Balanced, dynamic, heavy modern rock with strong drum weight.",
            },
            {
                "song": "Pneuma",
                "artist": "Tool",
                "source_url": "https://www.izotope.com/en/learn/12-great-reference-mixes-for-rock",
                "why": "Long-form loudness build, tight drums, and articulate heavy guitars.",
            },
        ],
        "tilt": {
            "weight_60_120_vs_plateau_db": 1.1,
            "sub_35_60_vs_plateau_db": -0.8,
            "high_4500_12000_vs_plateau_db": -1.5,
            "plateau_spread_db": 6.0,
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": 4.4,
            "lead_over_bgv_presence_db": 4.2,
            "kick_over_bass_55_95_db": 1.1,
        },
        "kick": {
            "click_share_in_master": 0.045,
            "click_share_in_drums": 0.14,
            "dynamic_range_max_db": 16.5,
            "click_minus_punch_db": -17.0,
            "box_minus_click_db": -1.5,
        },
        "style_summary": {
            "dynamic_range_db": 8.3,
            "stereo_width": 0.32,
            "crest_factor_db": 12.0,
        },
    },
}


@dataclass
class ChannelPlan:
    path: Path
    name: str
    instrument: str
    pan: float
    hpf: float
    target_rms_db: float
    lpf: float = 0.0
    trim_db: float = 0.0
    fader_db: float = 0.0
    muted: bool = False
    phase_invert: bool = False
    delay_ms: float = 0.0
    input_fade_ms: float = 0.0
    eq_bands: list[tuple[float, float, float]] = field(default_factory=list)
    comp_threshold_db: float = -20.0
    comp_ratio: float = 2.5
    comp_attack_ms: float = 10.0
    comp_release_ms: float = 120.0
    expander_enabled: bool = False
    expander_range_db: float = 0.0
    expander_open_ms: float = 12.0
    expander_close_ms: float = 140.0
    expander_hold_ms: float = 0.0
    expander_threshold_db: float | None = None
    expander_report: dict[str, Any] = field(default_factory=dict)
    event_activity: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    trim_analysis: dict[str, Any] = field(default_factory=dict)
    phase_notes: list[dict[str, Any]] = field(default_factory=list)
    pan_notes: list[dict[str, Any]] = field(default_factory=list)
    cross_adaptive_eq: list[dict[str, Any]] = field(default_factory=list)
    autofoh_actions: list[dict[str, Any]] = field(default_factory=list)
    fx_send_db: float | None = None
    fx_bus_send_db: dict[int, float] = field(default_factory=dict)


@dataclass
class LayerGroupPlan:
    group_id: str
    instrument: str
    channels: list[int]
    roles: dict[int, str]
    group_kind: str


@dataclass
class ReferenceMixContext:
    path: Path
    source_type: str
    style_profile: StyleProfile
    audio: np.ndarray | None = None
    sample_rate: int | None = None
    source_paths: list[Path] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    targets: dict[str, Any] = field(default_factory=dict)


def _agent_actions_to_dict(actions: list[AgentAction]) -> list[dict[str, Any]]:
    """Serialize action objects for report output."""
    return [action.__dict__ for action in actions]


class VirtualConsole:
    """MixerClient-like adapter for MixingAgent offline actions."""

    def __init__(self, plans: dict[int, ChannelPlan]):
        self.plans = plans
        self.calls: list[dict[str, Any]] = []

    def get_fader(self, channel: int) -> float:
        return self.plans[channel].fader_db

    def set_fader(self, channel: int, value_db: float):
        self.plans[channel].fader_db = float(np.clip(value_db, -30.0, 0.0))
        self.calls.append({"cmd": "set_fader", "channel": channel, "value_db": self.plans[channel].fader_db})

    def get_mute(self, channel: int) -> bool:
        return self.plans[channel].muted

    def set_mute(self, channel: int, muted: bool):
        self.plans[channel].muted = bool(muted)
        self.calls.append({"cmd": "set_mute", "channel": channel, "muted": bool(muted)})

    def set_hpf(self, channel: int, freq: float, enabled: bool = True):
        self.plans[channel].hpf = float(freq) if enabled else 0.0
        self.calls.append({"cmd": "set_hpf", "channel": channel, "freq": self.plans[channel].hpf})

    def set_polarity(self, channel: int, inverted: bool):
        self.plans[channel].phase_invert = bool(inverted)
        self.calls.append({"cmd": "set_polarity", "channel": channel, "inverted": bool(inverted)})

    def set_channel_phase_invert(self, channel: int, value: int):
        self.set_polarity(channel, bool(value))

    def set_delay(self, channel: int, delay_ms: float, enabled: bool = True):
        self.plans[channel].delay_ms = float(delay_ms) if enabled else 0.0
        self.calls.append({"cmd": "set_delay", "channel": channel, "delay_ms": self.plans[channel].delay_ms})

    def set_channel_delay(self, channel: int, value: float, mode: str = "MS"):
        self.set_delay(channel, float(value), enabled=True)

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        plan = self.plans[channel]
        band_idx = max(1, int(band)) - 1
        while len(plan.eq_bands) <= band_idx:
            plan.eq_bands.append((1000.0, 0.0, 1.0))
        plan.eq_bands[band_idx] = (float(freq), float(gain), float(q))
        self.calls.append({"cmd": "set_eq_band", "channel": channel, "band": band, "freq": freq, "gain": gain, "q": q})

    def set_compressor(self, channel: int, threshold_db: float, ratio: float, attack_ms: float, release_ms: float, enabled: bool = True):
        if not enabled:
            return
        plan = self.plans[channel]
        plan.comp_threshold_db = float(threshold_db)
        plan.comp_ratio = float(ratio)
        plan.comp_attack_ms = float(attack_ms)
        plan.comp_release_ms = float(release_ms)
        self.calls.append({
            "cmd": "set_compressor",
            "channel": channel,
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
        })


class OfflineAutoFOHConsole:
    """Minimal mixer adapter so offline analyzer actions use the same safety layer."""

    def __init__(self, plans: dict[int, ChannelPlan]):
        self.plans = plans
        self.calls: list[dict[str, Any]] = []

    def get_fader(self, channel: int) -> float:
        return float(self.plans[channel].fader_db)

    def set_fader(self, channel: int, value_db: float):
        plan = self.plans[channel]
        before = float(plan.fader_db)
        plan.fader_db = float(np.clip(value_db, -30.0, 0.0))
        self.calls.append({
            "cmd": "set_fader",
            "channel": channel,
            "before_db": round(before, 2),
            "after_db": round(plan.fader_db, 2),
        })

    def get_gain(self, channel: int) -> float:
        return float(self.plans[channel].trim_db)

    def set_gain(self, channel: int, value_db: float):
        plan = self.plans[channel]
        before = float(plan.trim_db)
        plan.trim_db = float(np.clip(value_db, -30.0, 12.0))
        self.calls.append({
            "cmd": "set_gain",
            "channel": channel,
            "before_db": round(before, 2),
            "after_db": round(plan.trim_db, 2),
        })

    @staticmethod
    def _band_index(band: int | str) -> int:
        token = str(band).strip().lower().rstrip("g")
        return max(1, int(token)) - 1

    def get_eq_band_gain(self, channel: int, band: int | str) -> float:
        plan = self.plans[channel]
        band_idx = self._band_index(band)
        if 0 <= band_idx < len(plan.eq_bands):
            return float(plan.eq_bands[band_idx][1])
        return 0.0

    def set_eq_band(self, channel: int, band: int, freq: float, gain: float, q: float):
        plan = self.plans[channel]
        band_idx = self._band_index(band)
        while len(plan.eq_bands) <= band_idx:
            plan.eq_bands.append((1000.0, 0.0, 1.0))
        before = plan.eq_bands[band_idx]
        plan.eq_bands[band_idx] = (float(freq), float(gain), float(q))
        self.calls.append({
            "cmd": "set_eq_band",
            "channel": channel,
            "band": band_idx + 1,
            "before": tuple(round(float(v), 2) for v in before),
            "after": (
                round(float(freq), 2),
                round(float(gain), 2),
                round(float(q), 2),
            ),
        })


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def amp_to_db(x: float) -> float:
    return float(20.0 * np.log10(max(float(x), 1e-12)))


def mono_sum(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio, axis=1).astype(np.float32)


def read_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return mono_sum(audio), int(sr)


def normalize_audio_shape(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim <= 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
        return arr.T.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Read audio with a small ffmpeg fallback so MP3 references work too."""
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return normalize_audio_shape(audio), int(sr)
    except Exception:
        with tempfile.TemporaryDirectory() as tmpdir:
            decoded = Path(tmpdir) / "decoded_reference.wav"
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(path),
                str(decoded),
            ]
            subprocess.run(cmd, check=True)
            audio, sr = sf.read(str(decoded), dtype="float32", always_2d=False)
            return normalize_audio_shape(audio), int(sr)


def resample_audio(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    arr = normalize_audio_shape(audio)
    if src_sr == dst_sr or len(arr) == 0:
        return arr.astype(np.float32, copy=True)

    dst_len = max(1, int(round(len(arr) * float(dst_sr) / float(src_sr))))
    src_pos = np.linspace(0.0, 1.0, num=len(arr), endpoint=False, dtype=np.float64)
    dst_pos = np.linspace(0.0, 1.0, num=dst_len, endpoint=False, dtype=np.float64)

    if arr.ndim == 1:
        return np.interp(dst_pos, src_pos, arr.astype(np.float64)).astype(np.float32)

    channels = [
        np.interp(dst_pos, src_pos, arr[:, idx].astype(np.float64)).astype(np.float32)
        for idx in range(arr.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32)


def _reference_channel_count(audio: np.ndarray) -> int:
    arr = normalize_audio_shape(audio)
    return 1 if arr.ndim == 1 else int(arr.shape[1])


def _match_reference_channels(audio: np.ndarray, channels: int) -> np.ndarray:
    arr = normalize_audio_shape(audio)
    if channels <= 1:
        return mono_sum(arr).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return np.column_stack([arr] * channels).astype(np.float32)
    if arr.shape[1] == channels:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] > channels:
        return arr[:, :channels].astype(np.float32, copy=False)
    last = arr[:, -1:]
    pads = [last] * (channels - arr.shape[1])
    return np.concatenate([arr, *pads], axis=1).astype(np.float32)


def _reference_excerpt(audio: np.ndarray, sr: int, max_duration_sec: float = 45.0) -> np.ndarray:
    arr = normalize_audio_shape(audio)
    max_samples = max(1, int(round(float(sr) * max_duration_sec)))
    if len(arr) <= max_samples:
        return arr.astype(np.float32, copy=True)
    start = max(0, (len(arr) - max_samples) // 2)
    return arr[start:start + max_samples].astype(np.float32, copy=True)


def _median_value(values: list[float], default: float = 0.0) -> float:
    usable: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            usable.append(numeric)
    if not usable:
        return float(default)
    return float(np.median(np.asarray(usable, dtype=np.float64)))


def _merge_style_profiles(style_profiles: list[StyleProfile], name: str) -> StyleProfile:
    if not style_profiles:
        raise ValueError("Cannot merge an empty style profile collection")
    if len(style_profiles) == 1:
        profile = style_profiles[0]
        return StyleProfile(
            name=name,
            spectral_balance=dict(profile.spectral_balance),
            dynamic_range=float(profile.dynamic_range),
            stereo_width=float(profile.stereo_width),
            loudness_lufs=float(profile.loudness_lufs),
            crest_factor=float(profile.crest_factor),
            per_instrument_settings=dict(profile.per_instrument_settings),
        )

    merged = StyleProfile(name=name)
    spectral_keys = sorted({
        band
        for profile in style_profiles
        for band in profile.spectral_balance.keys()
    })
    merged.spectral_balance = {
        band: round(
            _median_value(
                [profile.spectral_balance.get(band, -80.0) for profile in style_profiles],
                default=-80.0,
            ),
            4,
        )
        for band in spectral_keys
    }
    merged.dynamic_range = round(_median_value([p.dynamic_range for p in style_profiles], default=0.0), 4)
    merged.stereo_width = round(_median_value([p.stereo_width for p in style_profiles], default=0.0), 4)
    merged.loudness_lufs = round(_median_value([p.loudness_lufs for p in style_profiles], default=-14.0), 4)
    merged.crest_factor = round(_median_value([p.crest_factor for p in style_profiles], default=10.0), 4)
    if any(getattr(profile, "instrument_settings_mode", "absolute") == "relative" for profile in style_profiles):
        merged.instrument_settings_mode = "relative"

    instrument_names = sorted({
        instrument
        for profile in style_profiles
        for instrument in profile.per_instrument_settings.keys()
    })
    for instrument in instrument_names:
        variants = [
            profile.per_instrument_settings[instrument]
            for profile in style_profiles
            if instrument in profile.per_instrument_settings
        ]
        if not variants:
            continue
        merged.per_instrument_settings[instrument] = InstrumentStyle(
            instrument_type=variants[0].instrument_type,
            gain_db=round(_median_value([item.gain_db for item in variants], default=0.0), 4),
            eq_low_shelf_db=round(_median_value([item.eq_low_shelf_db for item in variants], default=0.0), 4),
            eq_low_mid_db=round(_median_value([item.eq_low_mid_db for item in variants], default=0.0), 4),
            eq_mid_db=round(_median_value([item.eq_mid_db for item in variants], default=0.0), 4),
            eq_high_mid_db=round(_median_value([item.eq_high_mid_db for item in variants], default=0.0), 4),
            eq_high_shelf_db=round(_median_value([item.eq_high_shelf_db for item in variants], default=0.0), 4),
            compression_ratio=round(_median_value([item.compression_ratio for item in variants], default=1.0), 4),
            compression_threshold_db=round(_median_value([item.compression_threshold_db for item in variants], default=-10.0), 4),
            gate_threshold_db=round(_median_value([item.gate_threshold_db for item in variants], default=-60.0), 4),
            pan=round(_median_value([item.pan for item in variants], default=0.0), 4),
            bus_send_level=round(_median_value([item.bus_send_level for item in variants], default=-96.0), 4),
        )
    return merged


def _iter_reference_sources(path: Path) -> list[Path]:
    candidates = [
        candidate.resolve()
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in (REFERENCE_AUDIO_SUFFIXES | REFERENCE_PRESET_SUFFIXES)
    ]
    if not candidates:
        raise FileNotFoundError(f"No supported reference files found in: {path}")
    return candidates


def _aggregate_reference_sections(
    audio: np.ndarray | None,
    sr: int | None,
    *,
    count: int = 4,
    window_sec: float = 12.0,
) -> list[dict[str, Any]]:
    if audio is None or sr is None or sr <= 0:
        return []

    mono = mono_sum(audio)
    if len(mono) == 0:
        return []

    actual_window_sec = float(min(window_sec, max(3.0, len(mono) / sr)))
    window = max(1024, int(actual_window_sec * sr))
    starts = sorted(set(_active_segment_starts(mono, sr, window_sec=actual_window_sec, count=count)))
    if not starts:
        starts = [0]

    sections: list[dict[str, Any]] = []
    meter = pyln.Meter(sr)
    for start in starts:
        end = min(len(mono), start + window)
        block = audio[start:end]
        if len(block) == 0:
            continue
        block_mono = mono[start:end]
        try:
            loudness = float(meter.integrated_loudness(_match_reference_channels(block, 2)))
        except Exception:
            loudness = float(StyleTransfer()._estimate_lufs(block_mono, sr))

        peak = float(np.max(np.abs(block))) if len(block) else 0.0
        rms = float(np.sqrt(np.mean(np.square(block_mono)))) if len(block_mono) else 0.0
        crest = amp_to_db(peak) - amp_to_db(rms) if rms > 1e-9 and peak > 1e-9 else 0.0
        sections.append({
            "start_sec": round(float(start / sr), 3),
            "end_sec": round(float(end / sr), 3),
            "lufs": round(loudness, 2),
            "crest_factor_db": round(float(crest), 2),
            "tilt": _ltas_tilt_profile(block, sr, window_sec=min(3.0, actual_window_sec), segments=3),
        })
    return sections


def _reference_target_pan(instrument: str, current_pan: float, width: float) -> float:
    width = float(np.clip(width, 0.0, 1.0))
    if instrument in {"lead_vocal", "kick", "snare", "bass_guitar", "bass", "bass_di", "bass_mic", "synth_bass"}:
        return 0.0
    if instrument in {"overhead", "oh_l", "oh_r", "room_l", "room_r"}:
        magnitude = float(np.clip(0.42 + width * 0.34, 0.32, 0.92))
    elif instrument in {"hi_hat", "ride", "electric_guitar", "acoustic_guitar", "keys", "strings", "backing_vocal"}:
        magnitude = float(np.clip(0.22 + width * 0.28, 0.12, 0.78))
    else:
        magnitude = float(np.clip(0.12 + width * 0.18, 0.0, 0.58))
    sign = -1.0 if current_pan < 0.0 else 1.0
    if abs(current_pan) < 0.05 and instrument in {"backing_vocal", "electric_guitar", "acoustic_guitar", "keys", "strings"}:
        sign = 1.0
    return sign * magnitude


def _reference_supports_expander(instrument: str) -> bool:
    return instrument not in {
        "overhead",
        "oh_l",
        "oh_r",
        "room",
        "room_l",
        "room_r",
        "cymbals",
        "reverb_return",
        "delay_return",
    }


def _normalize_genre_token(genre: str | None) -> str:
    token = str(genre or "").strip().lower()
    if not token:
        return ""
    token = token.replace("&", "and")
    token = re.sub(r"[^a-z0-9]+", "_", token).strip("_")
    return GENRE_REFERENCE_ALIASES.get(token, token)


def _genre_reference_seed(genre: str | None) -> dict[str, Any]:
    token = _normalize_genre_token(genre)
    if not token:
        return {}
    seed = GENRE_REFERENCE_STARTS.get(token)
    if not seed:
        return {}
    seeded = copy.deepcopy(seed)
    seeded["genre"] = token
    seeded["source_mode"] = "genre_seed_only"
    return seeded


def _merge_balance_targets(seed_targets: dict[str, Any], reference_targets: dict[str, Any]) -> dict[str, Any]:
    if not seed_targets:
        merged = copy.deepcopy(reference_targets)
        if merged and "source_mode" not in merged:
            merged["source_mode"] = "reference_only"
        return merged
    if not reference_targets:
        return copy.deepcopy(seed_targets)

    merged = copy.deepcopy(seed_targets)
    for key, value in reference_targets.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(copy.deepcopy(value))
        else:
            merged[key] = copy.deepcopy(value)
    merged["source_mode"] = "genre_seed_plus_reference"
    return merged


def _effective_balance_targets(
    reference_context: ReferenceMixContext | None,
    genre: str | None = None,
) -> dict[str, Any]:
    seed_targets = _genre_reference_seed(genre)
    if reference_context is None:
        return seed_targets
    reference_targets = _reference_targets_from_context(reference_context)
    if not reference_targets:
        return seed_targets
    return _merge_balance_targets(seed_targets, reference_targets)


def _reference_targets_from_context(reference_context: ReferenceMixContext | None) -> dict[str, Any]:
    if reference_context is None:
        return {}
    if reference_context.targets:
        return reference_context.targets

    style = reference_context.style_profile
    sections = _aggregate_reference_sections(reference_context.audio, reference_context.sample_rate)
    if sections:
        weight_target = float(np.median([item["tilt"]["weight_60_120_vs_plateau_db"] for item in sections]))
        sub_target = float(np.median([item["tilt"]["sub_35_60_vs_plateau_db"] for item in sections]))
        high_target = float(np.median([item["tilt"]["high_4500_12000_vs_plateau_db"] for item in sections]))
        plateau_target = float(np.median([item["tilt"]["plateau_spread_db"] for item in sections]))
    else:
        spectral = style.spectral_balance
        weight_target = float(np.clip((spectral.get("bass", -12.0) - spectral.get("mid", -11.5)) * 0.7 + 0.5, -1.0, 5.5))
        sub_target = float(np.clip((spectral.get("sub_bass", -24.0) - spectral.get("mid", -11.5)) * 0.5 - 0.5, -3.0, 3.0))
        high_target = float(np.clip((spectral.get("presence", -19.5) - spectral.get("mid", -11.5)) * 0.8 - 1.0, -6.0, 2.5))
        plateau_target = 5.5

    kick_style = style.per_instrument_settings.get("kick", InstrumentStyle("kick"))
    bass_style = style.per_instrument_settings.get("bass", InstrumentStyle("bass"))
    vocal_style = style.per_instrument_settings.get("vocals", InstrumentStyle("vocals"))
    kick_advantage = float(np.clip(
        0.9
        + (kick_style.gain_db - bass_style.gain_db) * 0.45
        + (kick_style.eq_low_shelf_db - bass_style.eq_low_shelf_db) * 0.30
        + kick_style.eq_high_mid_db * 0.18,
        0.5,
        2.8,
    ))
    lead_gap = float(np.clip(
        3.6
        + vocal_style.eq_high_mid_db * 0.35
        + vocal_style.gain_db * 0.45
        + max(0.0, 0.35 - style.stereo_width) * 1.6,
        2.8,
        6.5,
    ))
    click_share_master = float(np.clip(0.022 + max(0.0, kick_style.eq_high_mid_db) * 0.006, 0.02, 0.06))
    click_share_drums = float(np.clip(0.075 + max(0.0, kick_style.eq_high_mid_db) * 0.012, 0.07, 0.18))
    kick_dynamic_range_max = float(np.clip(17.5 - (kick_style.compression_ratio - 3.0) * 1.15, 11.0, 18.5))
    click_minus_punch = float(np.clip(-20.0 + kick_style.eq_high_mid_db * 2.3, -21.0, -12.5))
    box_minus_click = float(np.clip(0.5 - kick_style.eq_high_mid_db * 0.6, -2.5, 2.0))

    reference_context.sections = sections
    reference_context.targets = {
        "tilt": {
            "weight_60_120_vs_plateau_db": round(weight_target, 3),
            "sub_35_60_vs_plateau_db": round(sub_target, 3),
            "high_4500_12000_vs_plateau_db": round(high_target, 3),
            "plateau_spread_db": round(plateau_target, 3),
        },
        "hierarchy": {
            "lead_over_bgv_rms_db": round(lead_gap, 3),
            "kick_over_bass_55_95_db": round(kick_advantage, 3),
        },
        "kick": {
            "click_share_in_master": round(click_share_master, 4),
            "click_share_in_drums": round(click_share_drums, 4),
            "dynamic_range_max_db": round(kick_dynamic_range_max, 3),
            "click_minus_punch_db": round(click_minus_punch, 3),
            "box_minus_click_db": round(box_minus_click, 3),
        },
        "style_summary": {
            "dynamic_range_db": round(float(style.dynamic_range), 3),
            "stereo_width": round(float(style.stereo_width), 3),
            "crest_factor_db": round(float(style.crest_factor), 3),
        },
    }
    return reference_context.targets


def prepare_reference_mix_context(reference_path: str | Path | None) -> ReferenceMixContext | None:
    if not reference_path:
        return None

    path = Path(reference_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Reference track not found: {path}")

    transfer = StyleTransfer(fft_size=4096, hop_size=1024)
    if path.is_file() and path.suffix.lower() == ".json":
        style_profile = transfer.load_preset(str(path))
        context = ReferenceMixContext(
            path=path,
            source_type="style_preset",
            style_profile=style_profile,
            source_paths=[path],
        )
        _reference_targets_from_context(context)
        return context
    if path.is_file():
        audio, ref_sr = read_audio_file(path)
        style_profile = transfer.extract_style(audio, sr=ref_sr, name=path.stem)
        context = ReferenceMixContext(
            path=path,
            source_type="audio",
            style_profile=style_profile,
            audio=audio,
            sample_rate=ref_sr,
            source_paths=[path],
        )
        _reference_targets_from_context(context)
        return context

    source_paths = _iter_reference_sources(path)
    style_profiles: list[StyleProfile] = []
    audio_sources: list[tuple[np.ndarray, int]] = []
    has_audio = False
    has_preset = False

    for source in source_paths:
        if source.suffix.lower() in REFERENCE_PRESET_SUFFIXES:
            style_profiles.append(transfer.load_preset(str(source)))
            has_preset = True
            continue
        audio, ref_sr = read_audio_file(source)
        style_profiles.append(transfer.extract_style(audio, sr=ref_sr, name=source.stem))
        audio_sources.append((audio, ref_sr))
        has_audio = True

    if not style_profiles:
        raise FileNotFoundError(f"No usable reference profiles could be loaded from: {path}")

    merged_profile = _merge_style_profiles(style_profiles, name=path.name)
    combined_audio = None
    target_sr = None
    if audio_sources:
        target_sr = int(audio_sources[0][1])
        target_channels = max(_reference_channel_count(audio) for audio, _ in audio_sources)
        excerpts: list[np.ndarray] = []
        for audio, ref_sr in audio_sources:
            prepared = audio
            if ref_sr != target_sr:
                prepared = resample_audio(prepared, ref_sr, target_sr)
            prepared = _reference_excerpt(prepared, target_sr)
            prepared = _match_reference_channels(prepared, target_channels)
            peak = float(np.max(np.abs(prepared))) if len(prepared) else 0.0
            if peak > 1e-6:
                prepared = (prepared / peak * 0.92).astype(np.float32)
            excerpts.append(prepared.astype(np.float32, copy=False))
        if excerpts:
            combined_audio = np.concatenate(excerpts, axis=0).astype(np.float32, copy=False)

    if has_audio and has_preset:
        source_type = "mixed_reference_directory"
    elif has_audio:
        source_type = "audio_directory"
    else:
        source_type = "style_preset_directory"

    context = ReferenceMixContext(
        path=path,
        source_type=source_type,
        style_profile=merged_profile,
        audio=combined_audio,
        sample_rate=target_sr,
        source_paths=source_paths,
    )
    _reference_targets_from_context(context)
    return context


def _reference_style_instrument(instrument: str) -> str:
    return REFERENCE_STYLE_INSTRUMENTS.get(instrument, "other")


def _reference_fader_scale(instrument: str) -> float:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.22
    if instrument in {"kick", "snare", "bass", "bass_guitar", "bass_di", "bass_mic", "synth_bass"}:
        return 0.18
    if instrument in {"hi_hat", "ride", "overhead", "oh_l", "oh_r", "cymbals"}:
        return 0.12
    return 0.15


def _reference_eq_scale(instrument: str) -> float:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.25
    if instrument in {"hi_hat", "ride", "overhead", "oh_l", "oh_r", "cymbals"}:
        return 0.18
    return 0.22


def _reference_comp_scale(instrument: str) -> float:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 0.3
    if instrument in {"kick", "snare", "bass", "bass_guitar", "bass_di", "bass_mic", "synth_bass"}:
        return 0.25
    return 0.18


def _merge_reference_eq_adjustment(
    plan: ChannelPlan,
    *,
    freq: float,
    gain_db: float,
    q: float,
) -> dict[str, Any] | None:
    if abs(gain_db) < 0.15:
        return None

    target_idx = None
    best_distance = float("inf")
    for idx, (band_freq, _, _) in enumerate(plan.eq_bands):
        distance = abs(math.log2(max(freq, 20.0) / max(float(band_freq), 20.0)))
        if distance < 0.45 and distance < best_distance:
            best_distance = distance
            target_idx = idx

    if target_idx is None:
        band = (
            float(freq),
            float(np.clip(gain_db, -6.0, 6.0)),
            float(np.clip(q, 0.5, 2.5)),
        )
        plan.eq_bands.append(band)
        return {
            "mode": "append",
            "after": {
                "freq_hz": round(band[0], 2),
                "gain_db": round(band[1], 2),
                "q": round(band[2], 2),
            },
        }

    before = plan.eq_bands[target_idx]
    after = (
        float(before[0]),
        float(np.clip(before[1] + gain_db, -6.0, 6.0)),
        float(np.clip((before[2] + q) * 0.5, 0.5, 2.5)),
    )
    plan.eq_bands[target_idx] = after
    return {
        "mode": "merge",
        "before": {
            "freq_hz": round(float(before[0]), 2),
            "gain_db": round(float(before[1]), 2),
            "q": round(float(before[2]), 2),
        },
        "after": {
            "freq_hz": round(float(after[0]), 2),
            "gain_db": round(float(after[1]), 2),
            "q": round(float(after[2]), 2),
        },
    }


def apply_reference_mix_guidance(
    plans: dict[int, ChannelPlan],
    sr: int,
    reference_context: ReferenceMixContext | None,
) -> dict[str, Any]:
    if reference_context is None:
        return {"enabled": False, "reason": "no_reference_supplied"}

    transfer = StyleTransfer(fft_size=4096, hop_size=1024)
    channel_audios: dict[str, np.ndarray] = {}
    channel_types: dict[str, str] = {}
    skipped_channels: list[dict[str, Any]] = []

    for channel, plan in plans.items():
        try:
            mono, file_sr = read_mono(plan.path)
            if file_sr != sr:
                raise ValueError(f"sample rate mismatch {file_sr} != {sr}")
            analysis_audio, _ = _analysis_signal_for_metrics(mono, sr, plan.instrument)
            channel_audios[f"ch{channel}"] = analysis_audio.astype(np.float32, copy=False)
            channel_types[f"ch{channel}"] = _reference_style_instrument(plan.instrument)
        except Exception as exc:
            skipped_channels.append({
                "channel": channel,
                "file": plan.path.name,
                "reason": str(exc),
            })

    if not channel_audios:
        return {
            "enabled": False,
            "reason": "no_channels_available",
            "reference_path": str(reference_context.path),
            "source_type": reference_context.source_type,
            "reference_sources": [str(path) for path in reference_context.source_paths],
            "skipped_channels": skipped_channels,
        }

    mixing_params = transfer.apply_style(
        reference_context.style_profile,
        channel_audios,
        channel_types,
        sr=sr,
        blend_instrument_settings=reference_context.style_profile.instrument_settings_mode == "relative",
    )

    actions: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        params = mixing_params.get(f"ch{channel}")
        if not params:
            continue

        action = {
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "reference_type": channel_types.get(f"ch{channel}", "other"),
            "fader": None,
            "eq": [],
            "compressor": None,
            "pan": None,
            "expander": None,
            "fx_send": None,
        }

        fader_step = float(
            np.clip(
                float(params.get("fader_db", 0.0)) * _reference_fader_scale(plan.instrument),
                -1.5,
                1.5,
            )
        )
        if abs(fader_step) >= 0.12:
            before = float(plan.fader_db)
            plan.fader_db = float(np.clip(plan.fader_db + fader_step, -30.0, 0.0))
            action["fader"] = {
                "before_db": round(before, 2),
                "delta_db": round(plan.fader_db - before, 2),
                "after_db": round(plan.fader_db, 2),
            }

        style_width = float(reference_context.style_profile.stereo_width)
        target_pan = _reference_target_pan(
            plan.instrument,
            float(plan.pan),
            float(params.get("pan", style_width)),
        )
        if plan.instrument in {"lead_vocal", "kick", "snare", "bass_guitar", "bass", "bass_di", "bass_mic", "synth_bass"}:
            target_pan = 0.0
        if style_width < 0.18 and plan.instrument in {
            "backing_vocal",
            "electric_guitar",
            "acoustic_guitar",
            "guitar",
            "keys",
            "strings",
            "playback",
            "accordion",
            "overhead",
            "oh_l",
            "oh_r",
        }:
            pan_blend = 0.82 if abs(plan.pan) > 0.05 else 0.7
        else:
            pan_blend = 0.55 if abs(plan.pan) > 0.05 else 0.42
        desired_pan = float(np.clip(plan.pan + (target_pan - plan.pan) * pan_blend, -1.0, 1.0))
        if abs(desired_pan - float(plan.pan)) >= 0.04:
            before_pan = float(plan.pan)
            plan.pan = desired_pan
            action["pan"] = {
                "before": round(before_pan, 3),
                "target": round(target_pan, 3),
                "after": round(float(plan.pan), 3),
            }

        eq_candidates = sorted(
            params.get("eq_bands", []),
            key=lambda item: abs(float(item.get("gain_db", 0.0))),
            reverse=True,
        )[:3]
        eq_scale = _reference_eq_scale(plan.instrument)
        for band in eq_candidates:
            scaled_gain = float(np.clip(float(band.get("gain_db", 0.0)) * eq_scale, -1.25, 1.25))
            if abs(scaled_gain) < 0.2:
                continue
            q = 0.75 if band.get("type") in {"low_shelf", "high_shelf"} else float(band.get("q", 1.2))
            merged = _merge_reference_eq_adjustment(
                plan,
                freq=float(band.get("frequency", 1000.0)),
                gain_db=scaled_gain,
                q=q,
            )
            if merged is not None:
                merged["requested_gain_db"] = round(float(band.get("gain_db", 0.0)), 2)
                action["eq"].append(merged)

        if plan.instrument not in REFERENCE_COMPRESSOR_SKIP:
            comp = params.get("compression") or {}
            if comp:
                comp_scale = _reference_comp_scale(plan.instrument)
                ratio_before = float(plan.comp_ratio)
                threshold_before = float(plan.comp_threshold_db)
                ratio_step = float(np.clip((float(comp.get("ratio", ratio_before)) - ratio_before) * comp_scale, -0.6, 0.6))
                threshold_step = float(np.clip((float(comp.get("threshold_db", threshold_before)) - threshold_before) * comp_scale, -2.0, 2.0))
                if abs(ratio_step) >= 0.05 or abs(threshold_step) >= 0.1:
                    plan.comp_ratio = float(np.clip(ratio_before + ratio_step, 1.0, 8.0))
                    plan.comp_threshold_db = float(np.clip(threshold_before + threshold_step, -40.0, 0.0))
                    action["compressor"] = {
                        "ratio_before": round(ratio_before, 2),
                        "ratio_after": round(plan.comp_ratio, 2),
                        "threshold_before_db": round(threshold_before, 2),
                        "threshold_after_db": round(plan.comp_threshold_db, 2),
                    }

        gate_threshold = float(params.get("gate_threshold", -60.0))
        if _reference_supports_expander(plan.instrument) and gate_threshold > -59.0:
            before_enabled = bool(plan.expander_enabled)
            before_threshold = plan.expander_threshold_db
            before_range = float(plan.expander_range_db)
            target_threshold = float(np.clip(gate_threshold, -58.0, -32.0))
            range_db = 4.0 if plan.instrument in {"lead_vocal", "backing_vocal"} else 6.0
            if plan.instrument in {"kick", "snare", "toms", "rack_tom", "floor_tom"}:
                range_db = 8.0
            plan.expander_enabled = True
            plan.expander_threshold_db = target_threshold
            plan.expander_range_db = max(before_range, range_db)
            if plan.instrument in {"lead_vocal", "backing_vocal"}:
                plan.expander_open_ms = min(plan.expander_open_ms, 16.0)
                plan.expander_close_ms = max(plan.expander_close_ms, 180.0)
                plan.expander_hold_ms = max(plan.expander_hold_ms, 140.0)
            elif plan.instrument in {"kick", "snare", "toms", "rack_tom", "floor_tom"}:
                plan.expander_open_ms = min(plan.expander_open_ms, 8.0)
                plan.expander_close_ms = max(plan.expander_close_ms, 120.0)
                plan.expander_hold_ms = max(plan.expander_hold_ms, 40.0)
            if (not before_enabled) or before_threshold != plan.expander_threshold_db or abs(plan.expander_range_db - before_range) >= 0.1:
                action["expander"] = {
                    "enabled_before": before_enabled,
                    "threshold_before_db": None if before_threshold is None else round(float(before_threshold), 2),
                    "threshold_after_db": round(float(plan.expander_threshold_db), 2),
                    "range_before_db": round(before_range, 2),
                    "range_after_db": round(float(plan.expander_range_db), 2),
                }

        bus_send = params.get("bus_send") or {}
        if bus_send:
            send_levels = []
            bus_targets: dict[int, float] = {}
            for value in bus_send.values():
                try:
                    send_levels.append(float(value))
                except (TypeError, ValueError):
                    continue
            for bus_name, value in bus_send.items():
                try:
                    bus_id = int(bus_name)
                    bus_targets[bus_id] = float(np.clip(float(value), -96.0, 0.0))
                except (TypeError, ValueError):
                    continue
            if send_levels:
                target_send = float(np.clip(np.median(send_levels), -96.0, 0.0))
                before_send = plan.fx_send_db
                if before_send is None:
                    plan.fx_send_db = target_send
                else:
                    plan.fx_send_db = float(np.clip(before_send + (target_send - before_send) * 0.55, -96.0, 0.0))
                if before_send != plan.fx_send_db:
                    action["fx_send"] = {
                        "before_db": None if before_send is None else round(float(before_send), 2),
                        "target_db": round(target_send, 2),
                        "after_db": round(float(plan.fx_send_db), 2),
                    }
            if bus_targets:
                plan.fx_bus_send_db.update(bus_targets)

        if action["fader"] or action["eq"] or action["compressor"] or action["pan"] or action["expander"] or action["fx_send"]:
            actions.append(action)

    style = reference_context.style_profile
    targets = _reference_targets_from_context(reference_context)
    return {
        "enabled": True,
        "reference_path": str(reference_context.path),
        "source_type": reference_context.source_type,
        "reference_sources": [str(path) for path in reference_context.source_paths],
        "applied_channel_count": len(actions),
        "style_profile": {
            "name": style.name,
            "loudness_lufs": round(float(style.loudness_lufs), 2),
            "dynamic_range_db": round(float(style.dynamic_range), 2),
            "stereo_width": round(float(style.stereo_width), 3),
            "crest_factor_db": round(float(style.crest_factor), 2),
            "spectral_balance": {
                key: round(float(value), 2)
                for key, value in style.spectral_balance.items()
            },
            "instrument_settings_mode": style.instrument_settings_mode,
        },
        "targets": targets,
        "sections": reference_context.sections,
        "actions": actions,
        "skipped_channels": skipped_channels,
    }


def delay_signal(x: np.ndarray, sr: int, delay_ms: float) -> np.ndarray:
    delay_samples = int(round(max(0.0, delay_ms) * sr / 1000.0))
    if delay_samples <= 0:
        return x
    return np.pad(x, (delay_samples, 0))[:len(x)].astype(np.float32)


def apply_genre_mix_profile(
    plans: dict[int, ChannelPlan],
    genre: str | None,
) -> dict[str, Any]:
    genre_token = _normalize_genre_token(genre)
    if not genre_token:
        return {"enabled": False, "reason": "no_genre_supplied"}
    genre_seed = _genre_reference_seed(genre_token)
    if genre_token != "rock":
        return {
            "enabled": False,
            "requested_genre": genre_token,
            "reason": "seed_targets_only_genre",
            "genre_reference_seed": genre_seed,
            "notes": [
                "This genre currently uses curated internet-derived balance targets rather than a dedicated static channel-profile pass.",
                "Stem verification and other target-aware passes still pick up the genre seed as a starting point.",
            ],
        }

    actions: list[dict[str, Any]] = []
    snare_layers = [
        plan.path.name
        for plan in plans.values()
        if plan.instrument == "snare"
    ]

    for channel, plan in plans.items():
        changes: list[dict[str, Any]] = []

        if plan.instrument == "lead_vocal":
            before = {
                "trim_db": float(plan.trim_db),
                "target_rms_db": float(plan.target_rms_db),
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
                "attack_ms": float(plan.comp_attack_ms),
                "release_ms": float(plan.comp_release_ms),
            }
            plan.target_rms_db = max(plan.target_rms_db, -18.5)
            plan.trim_db = float(np.clip(plan.trim_db + 1.4, -18.0, 12.0))
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -26.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 4.8))
            plan.comp_attack_ms = float(min(plan.comp_attack_ms, 4.0))
            plan.comp_release_ms = float(np.clip(plan.comp_release_ms, 105.0, 125.0))
            changes.append({
                "type": "lead_vocal_glue",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "trim_db": round(float(plan.trim_db), 2),
                    "target_rms_db": round(float(plan.target_rms_db), 2),
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                    "attack_ms": round(float(plan.comp_attack_ms), 2),
                    "release_ms": round(float(plan.comp_release_ms), 2),
                },
            })
            if plan.expander_enabled:
                before_expander = {
                    "range_db": float(plan.expander_range_db),
                    "threshold_db": float(plan.expander_threshold_db or 0.0),
                    "open_ms": float(plan.expander_open_ms),
                    "close_ms": float(plan.expander_close_ms),
                    "hold_ms": float(plan.expander_hold_ms),
                }
                plan.expander_range_db = float(min(plan.expander_range_db, 3.2))
                if plan.expander_threshold_db is not None:
                    plan.expander_threshold_db = float(plan.expander_threshold_db - 1.0)
                plan.expander_open_ms = float(max(plan.expander_open_ms, 22.0))
                plan.expander_close_ms = float(max(plan.expander_close_ms, 240.0))
                plan.expander_hold_ms = float(max(plan.expander_hold_ms, 300.0))
                changes.append({
                    "type": "lead_vocal_phrase_stability",
                    "before": {key: round(value, 2) for key, value in before_expander.items()},
                    "after": {
                        "range_db": round(float(plan.expander_range_db), 2),
                        "threshold_db": round(float(plan.expander_threshold_db or 0.0), 2),
                        "open_ms": round(float(plan.expander_open_ms), 2),
                        "close_ms": round(float(plan.expander_close_ms), 2),
                        "hold_ms": round(float(plan.expander_hold_ms), 2),
                    },
                })

        elif plan.instrument == "backing_vocal":
            before = {
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
                "attack_ms": float(plan.comp_attack_ms),
                "release_ms": float(plan.comp_release_ms),
            }
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -24.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 3.3))
            plan.comp_attack_ms = float(min(plan.comp_attack_ms, 6.0))
            plan.comp_release_ms = float(np.clip(plan.comp_release_ms, 115.0, 145.0))
            changes.append({
                "type": "backing_vocal_tighten",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                    "attack_ms": round(float(plan.comp_attack_ms), 2),
                    "release_ms": round(float(plan.comp_release_ms), 2),
                },
            })

        elif plan.instrument == "snare":
            before = {
                "pan": float(plan.pan),
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
            }
            plan.pan = 0.0
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -23.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 4.0))
            changes.append({
                "type": "rock_snare_center",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "pan": round(float(plan.pan), 2),
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                },
            })

        elif plan.instrument == "kick":
            before = {
                "threshold_db": float(plan.comp_threshold_db),
                "ratio": float(plan.comp_ratio),
            }
            plan.comp_threshold_db = float(min(plan.comp_threshold_db, -20.0))
            plan.comp_ratio = float(max(plan.comp_ratio, 4.2))
            changes.append({
                "type": "rock_kick_anchor",
                "before": {key: round(value, 2) for key, value in before.items()},
                "after": {
                    "threshold_db": round(float(plan.comp_threshold_db), 2),
                    "ratio": round(float(plan.comp_ratio), 2),
                },
            })

        if changes:
            actions.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "changes": changes,
            })

    return {
        "enabled": bool(actions),
        "genre": genre_token,
        "genre_reference_seed": genre_seed,
        "notes": [
            "Rock mode tightens lead vocal glue with stronger compression and steadier between-phrase behavior.",
            "Snare layers stay treated as one center snare voice when both top and bottom mics are present.",
        ],
        "snare_layers": snare_layers,
        "actions": actions,
    }


def declick_start(x: np.ndarray, sr: int, fade_ms: float, threshold_db: float = -70.0) -> np.ndarray:
    if fade_ms <= 0.0 or len(x) == 0:
        return x
    threshold = db_to_amp(threshold_db)
    active = np.flatnonzero(np.abs(x) > threshold)
    if len(active) == 0:
        return x
    fade_len = max(16, int(fade_ms * sr / 1000.0))
    start = max(0, int(active[0]) - fade_len // 4)
    end = min(len(x), start + fade_len)
    if end <= start:
        return x
    out = x.copy()
    out[start:end] *= np.linspace(0.0, 1.0, end - start, dtype=np.float32)
    return out.astype(np.float32)


def _active_segment_start(x: np.ndarray, sr: int, window_sec: float = 3.0) -> int:
    window = max(1024, int(window_sec * sr))
    if len(x) <= window:
        return 0
    hop = max(512, window // 2)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return best_start


def _active_segment_starts(x: np.ndarray, sr: int, window_sec: float = 3.0, count: int = 8) -> list[int]:
    window = max(1024, int(window_sec * sr))
    if len(x) <= window:
        return [0]
    hop = max(512, window // 2)
    candidates = []
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        candidates.append((energy, start))
    candidates.sort(reverse=True)
    return [start for _, start in candidates[:count]]


def _merge_ranges(ranges: list[tuple[int, int]], gap: int = 0) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _event_metric_config(instrument: str) -> dict[str, float] | None:
    if instrument in {"lead_vocal", "backing_vocal"}:
        return {
            "frame_ms": 240.0,
            "hop_ms": 60.0,
            "pad_ms": 140.0,
            "detect_hpf_hz": 120.0,
            "detect_lpf_hz": 4500.0,
            "percentile": 82.0,
            "peak_offset_db": 18.0,
            "floor_margin_db": 8.0,
            "min_threshold_db": -42.0,
        }
    if instrument in {"rack_tom", "floor_tom"}:
        return {
            "frame_ms": 150.0,
            "hop_ms": 35.0,
            "pad_ms": 90.0,
            "detect_hpf_hz": 60.0,
            "detect_lpf_hz": 1200.0,
            "percentile": 97.5,
            "peak_offset_db": 14.0,
            "floor_margin_db": 10.0,
            "min_threshold_db": -36.0,
        }
    if instrument == "kick":
        return {
            "frame_ms": 140.0,
            "hop_ms": 30.0,
            "pad_ms": 80.0,
            "detect_hpf_hz": 28.0,
            "detect_lpf_hz": 220.0,
            "percentile": 95.0,
            "peak_offset_db": 14.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    if instrument == "snare":
        return {
            "frame_ms": 135.0,
            "hop_ms": 30.0,
            "pad_ms": 85.0,
            "detect_hpf_hz": 140.0,
            "detect_lpf_hz": 2600.0,
            "percentile": 97.0,
            "peak_offset_db": 13.0,
            "floor_margin_db": 9.0,
            "min_threshold_db": -38.0,
        }
    if instrument == "hi_hat":
        return {
            "frame_ms": 160.0,
            "hop_ms": 40.0,
            "pad_ms": 110.0,
            "detect_hpf_hz": 1800.0,
            "detect_lpf_hz": 14000.0,
            "percentile": 92.0,
            "peak_offset_db": 16.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    if instrument == "ride":
        return {
            "frame_ms": 170.0,
            "hop_ms": 40.0,
            "pad_ms": 120.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 94.0,
            "peak_offset_db": 18.0,
            "peak_percentile": 99.9,
            "floor_margin_db": 5.0,
            "min_threshold_db": -50.0,
        }
    if instrument == "percussion":
        return {
            "frame_ms": 160.0,
            "hop_ms": 40.0,
            "pad_ms": 110.0,
            "detect_hpf_hz": 1200.0,
            "detect_lpf_hz": 12000.0,
            "percentile": 92.0,
            "peak_offset_db": 16.0,
            "floor_margin_db": 6.0,
            "min_threshold_db": -44.0,
        }
    return None


def _next_power_of_two(value: int) -> int:
    return 1 << max(1, int(value - 1).bit_length())


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    aa = a[:n] - float(np.mean(a[:n]))
    bb = b[:n] - float(np.mean(b[:n]))
    denom = math.sqrt(float(np.dot(aa, aa) * np.dot(bb, bb))) + 1e-12
    return float(np.dot(aa, bb) / denom)


def _drum_channel_names(instrument: str) -> bool:
    return instrument in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}


def apply_drum_phase_alignment(plans: dict[int, ChannelPlan], sr: int, max_delay_ms: float = 12.0) -> list[dict[str, Any]]:
    """Apply the project's GCC-PHAT close-mic-to-overhead drum alignment rule."""
    overheads = [plan for plan in plans.values() if plan.instrument == "overhead"]
    if len(overheads) < 2:
        return []

    overhead_signals = []
    for plan in overheads:
        signal, file_sr = read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        overhead_signals.append(signal)
    reference = np.mean(np.vstack(overhead_signals), axis=0).astype(np.float32)
    reference = highpass(reference, sr, 40.0)

    reports: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        if not _drum_channel_names(plan.instrument):
            continue

        target, file_sr = read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        target = highpass(target, sr, 40.0)
        window_len = int(3.0 * sr)
        starts = _active_segment_starts(target, sr)
        measurements = []
        for start in starts:
            window = min(len(reference) - start, len(target) - start, window_len)
            if window < int(0.5 * sr):
                continue
            ref_seg = reference[start:start + window]
            tgt_seg = target[start:start + window]
            fft_size = _next_power_of_two(len(ref_seg))
            if fft_size != len(ref_seg):
                ref_seg = np.pad(ref_seg, (0, fft_size - len(ref_seg)))
                tgt_seg = np.pad(tgt_seg, (0, fft_size - len(tgt_seg)))

            analyzer = GCCPHATAnalyzer(sample_rate=sr, fft_size=fft_size, max_delay_ms=max_delay_ms)
            measurement = analyzer.compute_delay(ref_seg, tgt_seg)
            measured_delay_ms = float(measurement.delay_ms)
            boundary_hit = abs(abs(measured_delay_ms) - max_delay_ms) < 0.05
            if boundary_hit or not np.isfinite(measured_delay_ms):
                continue
            measurements.append((start, measurement))

        if not measurements:
            continue

        negative_measurements = [(start, m) for start, m in measurements if float(m.delay_ms) < -0.3]
        if negative_measurements:
            median_delay = float(np.median([float(m.delay_ms) for _, m in negative_measurements]))
            start, measurement = min(
                negative_measurements,
                key=lambda item: abs(float(item[1].delay_ms) - median_delay),
            )
        else:
            start, measurement = measurements[0]
        measured_delay_ms = float(measurement.delay_ms)

        delay_ms = 0.0
        if measured_delay_ms < -0.3:
            delay_ms = min(max_delay_ms, -measured_delay_ms)
            plan.delay_ms = max(plan.delay_ms, delay_ms)

        aligned = delay_signal(target, sr, plan.delay_ms)
        corr_start = start
        corr_end = min(corr_start + window_len, len(reference), len(aligned))
        corr_ref = reference[corr_start:corr_end]
        corr_target = aligned[corr_start:corr_end]
        current_corr = _norm_corr(corr_ref, -corr_target if plan.phase_invert else corr_target)
        flipped_corr = _norm_corr(corr_ref, corr_target if plan.phase_invert else -corr_target)
        if abs(flipped_corr) > 0.12 and flipped_corr > current_corr + 0.04:
            plan.phase_invert = not plan.phase_invert

        note = {
            "reference": "overhead_pair",
            "measured_delay_ms": round(measured_delay_ms, 3),
            "applied_delay_ms": round(plan.delay_ms, 3),
            "psr_db": round(float(measurement.psr), 2) if np.isfinite(measurement.psr) else None,
            "confidence": round(float(measurement.confidence), 3) if np.isfinite(measurement.confidence) else None,
            "coherence": round(float(measurement.coherence), 3) if np.isfinite(measurement.coherence) else None,
            "phase_invert": plan.phase_invert,
            "corr_current": round(current_corr, 3),
            "corr_flipped": round(flipped_corr, 3),
        }
        plan.phase_notes.append(note)
        reports.append({"channel": channel, "file": plan.path.name, **note})

    return reports


def _equal_power_gains(pan: float) -> tuple[float, float]:
    pan = float(np.clip(pan, -1.0, 1.0))
    theta = (pan + 1.0) * math.pi / 4.0
    return math.cos(theta), math.sin(theta)


def _pan_from_lr_diff_db(diff_db: float) -> float:
    ratio = db_to_amp(float(np.clip(diff_db, -48.0, 48.0)))
    theta = math.atan(ratio)
    return float(np.clip((4.0 * theta / math.pi) - 1.0, -1.0, 1.0))


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values, dtype=np.float64))
    sorted_values = np.asarray(values, dtype=np.float64)[order]
    sorted_weights = np.asarray(weights, dtype=np.float64)[order]
    cutoff = float(np.sum(sorted_weights)) * 0.5
    cumulative = np.cumsum(sorted_weights)
    idx = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def _transient_segment_starts(
    x: np.ndarray,
    sr: int,
    window_ms: float = 220.0,
    count: int = 16,
    min_gap_ms: float = 260.0,
) -> list[int]:
    """Find short drum events so overhead image placement is not averaged out."""
    window = max(256, int(window_ms * 0.001 * sr))
    if len(x) <= window:
        return [0]

    frame = max(128, int(0.018 * sr))
    hop = max(64, int(0.009 * sr))
    if len(x) <= frame:
        return [0]

    starts = np.arange(0, len(x) - frame, hop, dtype=np.int64)
    envelope = np.asarray([
        float(np.sqrt(np.mean(np.square(x[start:start + frame]))) + 1e-12)
        for start in starts
    ], dtype=np.float32)
    if len(envelope) < 3:
        return [0]

    floor = float(np.percentile(envelope, 60))
    threshold = max(floor * 1.4, float(np.percentile(envelope, 82)))
    peaks: list[tuple[float, int]] = []
    for idx in range(1, len(envelope) - 1):
        value = float(envelope[idx])
        if value < threshold:
            continue
        if value >= float(envelope[idx - 1]) and value >= float(envelope[idx + 1]):
            peaks.append((value, int(starts[idx] + frame // 2)))

    if not peaks:
        return _active_segment_starts(x, sr, window_sec=window_ms * 0.001, count=count)

    peaks.sort(reverse=True)
    min_gap = int(min_gap_ms * 0.001 * sr)
    selected: list[int] = []
    for _, center in peaks:
        if all(abs(center - existing) >= min_gap for existing in selected):
            selected.append(center)
        if len(selected) >= count:
            break

    selected.sort()
    return [int(np.clip(center - window // 2, 0, len(x) - window)) for center in selected]


def _overhead_lr(plans: dict[int, ChannelPlan]) -> tuple[tuple[int, ChannelPlan], tuple[int, ChannelPlan]] | None:
    overheads = [(channel, plan) for channel, plan in plans.items() if plan.instrument == "overhead"]
    if len(overheads) < 2:
        return None

    left = next(
        ((channel, plan) for channel, plan in overheads if "overhead l" in plan.name.lower() or plan.name.lower().endswith(" l")),
        None,
    )
    right = next(
        ((channel, plan) for channel, plan in overheads if "overhead r" in plan.name.lower() or plan.name.lower().endswith(" r")),
        None,
    )
    if left and right:
        return left, right

    ordered = sorted(overheads, key=lambda item: item[1].pan)
    return ordered[0], ordered[-1]


def _source_overhead_measurements(
    source: np.ndarray,
    overhead_l: np.ndarray,
    overhead_r: np.ndarray,
    sr: int,
    source_delay_ms: float,
    window_ms: float,
    count: int,
) -> list[dict[str, Any]]:
    window = max(512, int(window_ms * 0.001 * sr))
    delay_samples = int(round(max(0.0, source_delay_ms) * sr / 1000.0))
    starts = _transient_segment_starts(source, sr, window_ms=window_ms, count=count)
    measurements: list[dict[str, Any]] = []
    for start in starts:
        oh_start = start + delay_samples
        oh_end = min(oh_start + window, len(overhead_l), len(overhead_r))
        src_end = min(start + window, len(source))
        if oh_start < 0 or oh_end - oh_start < window // 3 or src_end - start < window // 3:
            continue

        src_seg = source[start:src_end]
        l_seg = overhead_l[oh_start:oh_end]
        r_seg = overhead_r[oh_start:oh_end]
        source_rms = float(np.sqrt(np.mean(np.square(src_seg))) + 1e-12)
        left_rms = float(np.sqrt(np.mean(np.square(l_seg))) + 1e-12)
        right_rms = float(np.sqrt(np.mean(np.square(r_seg))) + 1e-12)
        if source_rms < db_to_amp(-60.0) or (left_rms + right_rms) < db_to_amp(-64.0):
            continue
        measurements.append({
            "start_sec": round(start / sr, 3),
            "left_rms": left_rms,
            "right_rms": right_rms,
            "left_db": round(amp_to_db(left_rms), 2),
            "right_db": round(amp_to_db(right_rms), 2),
            "source_db": round(amp_to_db(source_rms), 2),
            "weight": float(np.clip(db_to_amp(amp_to_db(source_rms) + 34.0), 0.15, 4.0)),
        })
    return measurements


def _overhead_output_diff_db(left_rms: float, right_rms: float, left_pan: float, right_pan: float) -> float:
    left_to_l, left_to_r = _equal_power_gains(left_pan)
    right_to_l, right_to_r = _equal_power_gains(right_pan)
    out_l = max(1e-12, left_rms * left_to_l + right_rms * right_to_l)
    out_r = max(1e-12, left_rms * left_to_r + right_rms * right_to_r)
    return amp_to_db(out_r) - amp_to_db(out_l)


def _weighted_center_error(measurements: list[dict[str, Any]], left_pan: float, right_pan: float) -> tuple[float, float]:
    if not measurements:
        return 0.0, 0.0
    errors = []
    signed = []
    weights = []
    for measurement in measurements:
        diff = _overhead_output_diff_db(
            float(measurement["left_rms"]),
            float(measurement["right_rms"]),
            left_pan,
            right_pan,
        )
        weight = float(measurement["weight"])
        errors.append(abs(diff))
        signed.append(diff)
        weights.append(weight)
    return (
        float(np.average(np.asarray(errors), weights=np.asarray(weights))),
        float(np.average(np.asarray(signed), weights=np.asarray(weights))),
    )


def apply_overhead_anchored_drum_panning(plans: dict[int, ChannelPlan], sr: int) -> dict[str, Any]:
    """Pan drums from the overhead picture: center kick/snare first, then place close mics."""
    pair = _overhead_lr(plans)
    if not pair:
        return {"enabled": False, "reason": "overhead_pair_not_found"}

    (left_channel, left_plan), (right_channel, right_plan) = pair
    overhead_l, file_sr = read_mono(left_plan.path)
    if file_sr != sr:
        raise ValueError(f"{left_plan.path.name}: sample rate mismatch {file_sr} != {sr}")
    overhead_r, file_sr = read_mono(right_plan.path)
    if file_sr != sr:
        raise ValueError(f"{right_plan.path.name}: sample rate mismatch {file_sr} != {sr}")
    overhead_l = highpass(overhead_l, sr, 45.0)
    overhead_r = highpass(overhead_r, sr, 45.0)

    center_measurements: list[dict[str, Any]] = []
    source_measurements: dict[int, list[dict[str, Any]]] = {}
    for channel, plan in plans.items():
        if plan.instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue
        source, file_sr = read_mono(plan.path)
        if file_sr != sr:
            raise ValueError(f"{plan.path.name}: sample rate mismatch {file_sr} != {sr}")
        source = highpass(source, sr, 40.0)
        window_ms = 170.0 if plan.instrument in {"kick", "snare"} else 260.0
        count = 18 if plan.instrument in {"kick", "snare"} else 12
        measurements = _source_overhead_measurements(
            source,
            overhead_l,
            overhead_r,
            sr,
            plan.delay_ms,
            window_ms=window_ms,
            count=count,
        )
        source_measurements[channel] = measurements
        if plan.instrument in {"kick", "snare"}:
            center_measurements.extend([
                {**measurement, "channel": channel, "file": plan.path.name, "instrument": plan.instrument}
                for measurement in measurements
            ])

    before_left = float(left_plan.pan)
    before_right = float(right_plan.pan)
    if center_measurements:
        best: tuple[float, float, float, float, float] | None = None
        for width in np.linspace(0.58, 0.88, 31):
            for shift in np.linspace(-0.18, 0.18, 37):
                left_pan = float(np.clip(-width + shift, -0.95, -0.18))
                right_pan = float(np.clip(width + shift, 0.18, 0.95))
                error, signed = _weighted_center_error(center_measurements, left_pan, right_pan)
                width_penalty = max(0.0, 0.74 - width) * 0.7 + max(0.0, width - 0.86) * 0.25
                score = error + width_penalty + abs(float(shift)) * 0.2
                if best is None or score < best[0]:
                    best = (score, left_pan, right_pan, error, signed)
        assert best is not None
        _, new_left, new_right, after_error, after_signed = best
        left_plan.pan = new_left
        right_plan.pan = new_right
    else:
        after_error, after_signed = _weighted_center_error(center_measurements, before_left, before_right)

    before_error, before_signed = _weighted_center_error(center_measurements, before_left, before_right)
    left_note = {
        "rule": "overhead_anchor_first",
        "before_pan": round(before_left, 3),
        "after_pan": round(left_plan.pan, 3),
        "center_sources": ["kick", "snare"],
        "center_error_before_db": round(before_error, 2),
        "center_error_after_db": round(after_error, 2),
        "center_signed_after_db": round(after_signed, 2),
    }
    right_note = dict(left_note)
    right_note["before_pan"] = round(before_right, 3)
    right_note["after_pan"] = round(right_plan.pan, 3)
    left_plan.pan_notes.append(left_note)
    right_plan.pan_notes.append(right_note)

    placed: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        if plan.instrument not in {"kick", "snare", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
            continue

        before_pan = float(plan.pan)
        if plan.instrument in {"kick", "snare"}:
            plan.pan = 0.0
            note = {
                "rule": "kick_snare_center_after_overheads",
                "before_pan": round(before_pan, 3),
                "after_pan": round(plan.pan, 3),
                "reason": "kick/snare image is the center reference for the overhead pair",
            }
            plan.pan_notes.append(note)
            placed.append({"channel": channel, "file": plan.path.name, "instrument": plan.instrument, **note})
            continue

        measurements = source_measurements.get(channel, [])
        pan_values = []
        weights = []
        for measurement in measurements:
            diff = _overhead_output_diff_db(
                float(measurement["left_rms"]),
                float(measurement["right_rms"]),
                left_plan.pan,
                right_plan.pan,
            )
            pan_values.append(_pan_from_lr_diff_db(diff))
            weights.append(float(measurement["weight"]))

        if pan_values:
            estimated_pan = _weighted_median(pan_values, weights)
            if plan.instrument in {"rack_tom", "floor_tom"}:
                estimated_pan = float(np.clip(estimated_pan, -0.68, 0.68))
            else:
                estimated_pan = float(np.clip(estimated_pan, -0.82, 0.82))
            plan.pan = estimated_pan
            source_note = {
                "rule": "close_mic_follows_overhead_image",
                "before_pan": round(before_pan, 3),
                "after_pan": round(plan.pan, 3),
                "measurement_count": len(measurements),
                "median_overhead_image_pan": round(estimated_pan, 3),
                "left_right_db_examples": [
                    {
                        "start_sec": measurement["start_sec"],
                        "left_db": measurement["left_db"],
                        "right_db": measurement["right_db"],
                    }
                    for measurement in measurements[:4]
                ],
            }
        else:
            source_note = {
                "rule": "close_mic_follows_overhead_image",
                "before_pan": round(before_pan, 3),
                "after_pan": round(plan.pan, 3),
                "measurement_count": 0,
                "reason": "no confident overhead image measurements, kept existing pan",
            }
        plan.pan_notes.append(source_note)
        placed.append({"channel": channel, "file": plan.path.name, "instrument": plan.instrument, **source_note})

    return {
        "enabled": True,
        "overhead_left_channel": left_channel,
        "overhead_right_channel": right_channel,
        "overhead_left_file": left_plan.path.name,
        "overhead_right_file": right_plan.path.name,
        "overhead_left_pan": round(left_plan.pan, 3),
        "overhead_right_pan": round(right_plan.pan, 3),
        "center_measurements": len(center_measurements),
        "center_error_before_db": round(before_error, 2),
        "center_error_after_db": round(after_error, 2),
        "center_signed_before_db": round(before_signed, 2),
        "center_signed_after_db": round(after_signed, 2),
        "placed_close_mics": placed,
    }


def _post_fader_lufs(plan: ChannelPlan) -> float:
    return float(plan.metrics["lufs_momentary"] + plan.trim_db + plan.fader_db)


def _music_bed_lufs(plans: dict[int, ChannelPlan]) -> float:
    music_reference = [
        _post_fader_lufs(plan)
        for plan in plans.values()
        if plan.instrument != "lead_vocal" and not plan.muted
    ]
    top_music = sorted(music_reference, reverse=True)[:5]
    return float(np.mean(top_music)) if top_music else -20.0


LEGACY_PRESET_BY_INSTRUMENT = {
    "kick": "kick",
    "snare": "snare",
    "rack_tom": "tom",
    "floor_tom": "tom",
    "hi_hat": "hihat",
    "ride": "ride",
    "overhead": "overheads",
    "room": "room",
    "bass_guitar": "bass",
    "electric_guitar": "electricGuitar",
    "acoustic_guitar": "acousticGuitar",
    "accordion": "accordion",
    "synth": "synth",
    "playback": "playback",
    "lead_vocal": "leadVocal",
    "backing_vocal": "backVocal",
    "custom": "custom",
}


def _legacy_preset_for_instrument(instrument: str) -> str:
    return LEGACY_PRESET_BY_INSTRUMENT.get(str(instrument or "").strip().lower(), "custom")


def _measurement_window_for_instrument(instrument: str) -> float:
    if instrument in {"kick", "snare", "rack_tom", "floor_tom"}:
        return 0.55
    if instrument in {"hi_hat", "ride", "percussion"}:
        return 0.8
    if instrument in {"lead_vocal", "backing_vocal"}:
        return 1.2
    if instrument in {"bass_guitar", "electric_guitar", "acoustic_guitar", "accordion", "playback", "synth"}:
        return 1.5
    return 1.0


def _focused_fft_measurement_block(
    x: np.ndarray,
    sr: int,
    instrument: str,
    *,
    fft_size: int = 4096,
) -> tuple[np.ndarray, dict[str, Any]]:
    analysis_signal, analysis_meta = _analysis_signal_for_metrics(x, sr, instrument)
    search_window = _analysis_block(
        analysis_signal,
        sr,
        window_sec=_measurement_window_for_instrument(instrument),
    )
    if len(search_window) == 0:
        return search_window.astype(np.float32), {
            **analysis_meta,
            "measurement_block_sec": 0.0,
            "measurement_block_samples": 0,
        }

    if len(search_window) <= fft_size:
        block = search_window
    else:
        hop = max(256, fft_size // 4)
        best_start = 0
        best_energy = -1.0
        for start in range(0, len(search_window) - fft_size + 1, hop):
            energy = float(np.mean(np.square(search_window[start:start + fft_size])))
            if energy > best_energy:
                best_energy = energy
                best_start = start
        block = search_window[best_start:best_start + fft_size]

    return block.astype(np.float32), {
        **analysis_meta,
        "measurement_block_sec": round(len(block) / sr, 4) if sr else 0.0,
        "measurement_block_samples": int(len(block)),
    }


def _typed_action_to_dict(action: TypedCorrectionAction) -> dict[str, Any]:
    payload = {
        "action_type": action.action_type,
        "reason": action.reason,
    }
    payload.update(action.__dict__)
    return payload


def _problem_to_dict(problem: Any) -> dict[str, Any]:
    if problem is None:
        return {}
    confidence = getattr(problem, "confidence_risk", None)
    return {
        "problem_type": getattr(problem, "problem_type", ""),
        "description": getattr(problem, "description", ""),
        "channel_id": getattr(problem, "channel_id", None),
        "stem": getattr(problem, "stem", None),
        "band_name": getattr(problem, "band_name", None),
        "persistence_sec": getattr(problem, "persistence_sec", 0.0),
        "expected_effect": getattr(problem, "expected_effect", ""),
        "confidence_risk": {
            "problem_confidence": round(float(getattr(confidence, "problem_confidence", 0.0)), 3),
            "culprit_confidence": round(float(getattr(confidence, "culprit_confidence", 0.0)), 3),
            "action_confidence": round(float(getattr(confidence, "action_confidence", 0.0)), 3),
            "risk_score": round(float(getattr(confidence, "risk_score", 0.0)), 3),
        },
    }


def _build_autofoh_measurement_snapshot(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    autofoh_config: dict[str, Any] | None = None,
):
    autofoh_config = autofoh_config or {}
    analysis_config = autofoh_config.get("analysis", {})
    fft_size = int(analysis_config.get("fft_size", 4096))
    octave_fraction = int(analysis_config.get("octave_fraction", 3))
    slope_db = float(analysis_config.get("slope_compensation_db_per_octave", 4.5))

    channel_features = {}
    channel_stems = {}
    channel_priorities = {}
    channel_measurements = {}

    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        mono = mono_sum(audio)
        block, measurement_meta = _focused_fft_measurement_block(
            mono,
            sr,
            plan.instrument,
            fft_size=fft_size,
        )
        if len(block) == 0:
            continue
        features = extract_analysis_features(
            block,
            sample_rate=sr,
            fft_size=fft_size,
            octave_fraction=octave_fraction,
            slope_compensation_db_per_octave=slope_db,
        )
        if features.confidence <= 0.0:
            continue

        classification = classification_from_legacy_preset(
            _legacy_preset_for_instrument(plan.instrument),
            channel_name=plan.name,
            confidence=1.0,
            match_type="offline_measurement",
        )
        stems = [stem.value for stem in classification.stem_roles if stem.value != "MASTER"]
        if not stems:
            stems = ["LEAD"] if plan.instrument == "lead_vocal" else ["UNKNOWN"]

        channel_features[channel] = features
        channel_stems[channel] = stems
        channel_priorities[channel] = float(classification.priority)
        channel_measurements[channel] = {
            "file": plan.path.name,
            "instrument": plan.instrument,
            "source_role": classification.source_role.value,
            "stem_roles": stems,
            "priority": round(float(classification.priority), 3),
            **measurement_meta,
        }

    if not channel_features:
        return {}, {}, None, {}, {}, {}

    stem_features = aggregate_stem_features(channel_features, channel_stems)
    contribution_matrix = build_stem_contribution_matrix(
        {
            stem_name: features
            for stem_name, features in stem_features.items()
            if stem_name != "MASTER"
        }
    )
    return (
        channel_features,
        stem_features,
        contribution_matrix,
        channel_stems,
        channel_priorities,
        channel_measurements,
    )


def _measured_channel_level_db(plan: ChannelPlan, features: Any) -> float:
    return float(getattr(features, "rms_db", -100.0) + plan.trim_db + plan.fader_db)


def _lead_handoff_balance_recommendations(
    plans: dict[int, ChannelPlan],
    channel_features: dict[int, Any],
    lead_channels: list[int],
) -> list[tuple[str, Any, list[TypedCorrectionAction]]]:
    measured_leads: list[tuple[int, float, float]] = []
    for channel in lead_channels:
        plan = plans.get(channel)
        features = channel_features.get(channel)
        if plan is None or features is None or plan.muted:
            continue
        active_ratio = float(plan.metrics.get("analysis_active_ratio") or 0.0)
        if active_ratio < 0.015:
            continue
        measured_leads.append((channel, _measured_channel_level_db(plan, features), active_ratio))

    if len(measured_leads) < 2:
        return []

    anchor_channel, anchor_level_db, _ = max(measured_leads, key=lambda item: item[1])
    anchor_target_db = anchor_level_db - 0.8
    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []

    for channel, level_db, active_ratio in measured_leads:
        if channel == anchor_channel:
            continue
        shortfall_db = anchor_target_db - level_db
        if shortfall_db < 0.75:
            continue

        plan = plans[channel]
        boost_db = float(min(1.5, shortfall_db))
        if plan.fader_db <= -0.25:
            action: TypedCorrectionAction = ChannelFaderMove(
                channel_id=channel,
                target_db=min(0.0, float(plan.fader_db + boost_db)),
                delta_db=boost_db,
                is_lead=True,
                reason=f"Measured lead handoff balance vs {plans[anchor_channel].path.name}",
            )
        else:
            action = ChannelGainMove(
                channel_id=channel,
                target_db=float(np.clip(plan.trim_db + boost_db, -12.0, 12.0)),
                reason=f"Measured lead handoff balance vs {plans[anchor_channel].path.name}",
            )

        confidence = min(1.0, 0.55 + shortfall_db / 3.0)
        recommendations.append((
            "lead_handoff_balance",
            DetectedProblem(
                problem_type="lead_handoff_balance",
                description="Measured secondary lead sits below the anchor lead level",
                channel_id=channel,
                stem="LEAD",
                band_name="RMS",
                persistence_sec=max(0.5, float(active_ratio * 10.0)),
                features=channel_features[channel],
                confidence_risk=ConfidenceRisk(
                    problem_confidence=confidence,
                    culprit_confidence=1.0,
                    action_confidence=min(1.0, confidence * 0.92),
                    risk_score=0.18,
                ),
                expected_effect="Bring quieter lead handoffs closer to the anchor lead without flattening the vocal hierarchy.",
            ),
            [action],
        ))

    return recommendations


def _lead_background_hierarchy_recommendations(
    plans: dict[int, ChannelPlan],
    channel_features: dict[int, Any],
    lead_channels: list[int],
) -> list[tuple[str, Any, list[TypedCorrectionAction]]]:
    if not lead_channels:
        return []

    backing_channels = [
        channel
        for channel, features in channel_features.items()
        if channel in plans and not plans[channel].muted and plans[channel].instrument == "backing_vocal"
    ]
    if not backing_channels:
        return []

    anchor_channel = max(
        lead_channels,
        key=lambda channel: (
            float(channel_features[channel].named_band_levels_db.get("PRESENCE", -100.0)),
            float(channel_features[channel].rms_db),
        ),
    )
    anchor_features = channel_features[anchor_channel]
    anchor_presence_db = float(anchor_features.named_band_levels_db.get("PRESENCE", -100.0))
    anchor_rms_db = float(anchor_features.rms_db)
    target_presence_gap_db = 4.0
    target_rms_gap_db = 5.5

    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []
    for channel in backing_channels:
        features = channel_features[channel]
        plan = plans[channel]
        bgv_presence_db = float(features.named_band_levels_db.get("PRESENCE", -100.0))
        bgv_rms_db = float(features.rms_db)
        presence_gap_db = anchor_presence_db - bgv_presence_db
        rms_gap_db = anchor_rms_db - bgv_rms_db
        shortfall_db = max(
            target_presence_gap_db - presence_gap_db,
            target_rms_gap_db - rms_gap_db,
        )
        if shortfall_db <= 0.9:
            continue

        cut_db = float(np.clip(0.8 + shortfall_db * 0.45, 0.8, 2.4))
        target_db = max(-6.0, float(plan.fader_db - cut_db))
        confidence = min(1.0, 0.58 + shortfall_db / 4.0)
        recommendations.append((
            "lead_background_hierarchy",
            DetectedProblem(
                problem_type="lead_background_hierarchy",
                description="Measured backing vocal sits too close to the lead vocal foreground",
                channel_id=channel,
                stem="BGV",
                band_name="PRESENCE",
                persistence_sec=max(1.0, float(plan.metrics.get("analysis_active_ratio") or 0.0) * 8.0),
                features=features,
                confidence_risk=ConfidenceRisk(
                    problem_confidence=confidence,
                    culprit_confidence=min(1.0, 0.55 + shortfall_db / 5.0),
                    action_confidence=min(1.0, 0.62 + shortfall_db / 5.0),
                    risk_score=0.16,
                ),
                expected_effect="Pull backing vocals behind the lead vocal image instead of letting the stack flatten the vocal hierarchy.",
            ),
            [ChannelFaderMove(
                channel_id=channel,
                target_db=target_db,
                delta_db=target_db - float(plan.fader_db),
                is_lead=False,
                reason=f"Measured lead-vs-BGV hierarchy on {plan.path.name}",
            )],
        ))

    return recommendations


def _lead_foreground_balance_recommendations(
    plans: dict[int, ChannelPlan],
    channel_features: dict[int, Any],
    stem_features: dict[str, Any],
    lead_channels: list[int],
) -> list[tuple[str, Any, list[TypedCorrectionAction]]]:
    if not lead_channels:
        return []

    lead_features = stem_features.get("LEAD")
    if lead_features is None:
        return []

    accompaniment_presence_power = 0.0
    for stem_name, feature in stem_features.items():
        if stem_name in {"MASTER", "LEAD"}:
            continue
        accompaniment_presence_power += 10.0 ** (
            float(feature.named_band_levels_db.get("PRESENCE", -100.0)) / 10.0
        )
    accompaniment_presence_db = 10.0 * np.log10(accompaniment_presence_power + 1e-10)
    lead_presence_db = float(lead_features.named_band_levels_db.get("PRESENCE", -100.0))
    shortfall_db = accompaniment_presence_db - lead_presence_db - 2.5
    if shortfall_db <= 0.9:
        return []

    boost_db = float(np.clip(0.45 + shortfall_db * 0.16, 0.45, 1.1))
    confidence = min(1.0, 0.58 + shortfall_db / 6.0)
    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []
    for channel in lead_channels:
        plan = plans.get(channel)
        features = channel_features.get(channel)
        if plan is None or features is None or plan.muted:
            continue
        action: TypedCorrectionAction
        if plan.fader_db <= -0.25:
            action = ChannelFaderMove(
                channel_id=channel,
                target_db=min(0.0, float(plan.fader_db + boost_db)),
                delta_db=boost_db,
                is_lead=True,
                reason=f"Measured lead foreground balance on {plan.path.name}",
            )
        else:
            action = ChannelGainMove(
                channel_id=channel,
                target_db=float(np.clip(plan.trim_db + boost_db, -12.0, 12.0)),
                reason=f"Measured lead foreground balance on {plan.path.name}",
            )
        recommendations.append((
            "lead_foreground_balance",
            DetectedProblem(
                problem_type="lead_foreground_balance",
                description="Measured lead stem still sits behind the accompaniment foreground",
                channel_id=channel,
                stem="LEAD",
                band_name="PRESENCE",
                persistence_sec=max(1.0, float(plan.metrics.get("analysis_active_ratio") or 0.0) * 10.0),
                features=features,
                confidence_risk=ConfidenceRisk(
                    problem_confidence=confidence,
                    culprit_confidence=1.0,
                    action_confidence=min(1.0, confidence * 0.95),
                    risk_score=0.18,
                ),
                expected_effect="Lift the lead stem back in front of the accompaniment without flattening internal lead handoffs.",
            ),
            [action],
        ))

    return recommendations


def _cymbal_buildup_recommendations(
    plans: dict[int, ChannelPlan],
    channel_features: dict[int, Any],
    master_features: Any | None,
) -> list[tuple[str, Any, list[TypedCorrectionAction]]]:
    if master_features is None:
        return []

    harshness_excess = max(0.0, float(master_features.mix_indexes.harshness_index))
    sibilance_excess = max(0.0, float(master_features.mix_indexes.sibilance_index))
    if max(harshness_excess, sibilance_excess) < 1.75:
        return []

    direct_candidates: list[tuple[int, float]] = []
    ambient_candidates: list[tuple[int, float]] = []
    for channel, features in channel_features.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        if plan.instrument not in {"hi_hat", "ride", "overhead", "room"}:
            continue
        score = (
            max(0.0, float(features.mix_indexes.harshness_index)) * 0.8
            + max(0.0, float(features.mix_indexes.sibilance_index)) * 1.2
            + max(0.0, float(features.mix_indexes.air_index)) * 0.4
        )
        if plan.instrument in {"hi_hat", "ride"}:
            score *= 1.15
            direct_candidates.append((channel, score))
        else:
            ambient_candidates.append((channel, score))

    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []
    dominant_band = "SIBILANCE" if sibilance_excess >= harshness_excess else "HARSHNESS"
    dominant_freq = 7500.0 if dominant_band == "SIBILANCE" else 4200.0
    dominant_q = 2.0 if dominant_band == "SIBILANCE" else 1.8

    for direct_channel, direct_score in sorted(direct_candidates, key=lambda item: item[1], reverse=True)[:2]:
        if direct_score >= 0.65:
            plan = plans[direct_channel]
            cut_db = float(min(2.25, 0.9 + 0.28 * max(harshness_excess, sibilance_excess)))
            target_db = max(-30.0, float(plan.fader_db - cut_db))
            recommendations.append((
                "cymbal_buildup",
                DetectedProblem(
                    problem_type="cymbal_buildup",
                    description="Measured direct cymbal energy is dominating the upper bands",
                    channel_id=direct_channel,
                    stem="DRUMS",
                    band_name=dominant_band,
                    persistence_sec=max(1.0, direct_score),
                    features=channel_features[direct_channel],
                    confidence_risk=ConfidenceRisk(
                        problem_confidence=min(1.0, 0.5 + direct_score / 3.0),
                        culprit_confidence=min(1.0, direct_score / 3.0),
                        action_confidence=0.82,
                        risk_score=0.22,
                    ),
                    expected_effect="Reduce direct cymbal dominance while preserving the drum image.",
                ),
                [ChannelFaderMove(
                    channel_id=direct_channel,
                    target_db=target_db,
                    delta_db=target_db - float(plan.fader_db),
                    reason=f"Measured cymbal buildup on {plan.path.name}",
                )],
            ))

    for ambient_channel, ambient_score in sorted(ambient_candidates, key=lambda item: item[1], reverse=True)[:2]:
        if ambient_score >= 0.52:
            plan = plans[ambient_channel]
            ambient_cut_db = float(min(0.9, 0.45 + ambient_score * 0.08))
            recommendations.append((
                "cymbal_buildup",
                DetectedProblem(
                    problem_type="cymbal_buildup",
                    description="Measured overhead or room cymbal wash is dominating the upper bands",
                    channel_id=ambient_channel,
                    stem="DRUMS",
                    band_name=dominant_band,
                    persistence_sec=max(1.0, ambient_score),
                    features=channel_features[ambient_channel],
                    confidence_risk=ConfidenceRisk(
                        problem_confidence=min(1.0, 0.48 + ambient_score / 3.5),
                        culprit_confidence=min(1.0, ambient_score / 3.5),
                        action_confidence=0.76,
                        risk_score=0.24,
                    ),
                    expected_effect="Reduce cymbal wash in the ambient drum capture without collapsing width.",
                ),
                [ChannelFaderMove(
                    channel_id=ambient_channel,
                    target_db=max(-6.0, float(plan.fader_db - ambient_cut_db)),
                    delta_db=-ambient_cut_db,
                    is_lead=False,
                    reason=f"Measured cymbal wash cleanup on {plan.path.name}",
                )],
            ))

    return recommendations


def apply_autofoh_measurement_corrections(
    plans: dict[int, ChannelPlan],
    rendered_channels: dict[int, np.ndarray],
    sr: int,
    autofoh_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    autofoh_config = autofoh_config or {}
    detector_config = autofoh_config.get("detectors", {})
    safety_config = AutoFOHSafetyConfig.from_config(
        autofoh_config.get("safety", {}).get("action_limits", {})
    )
    (
        channel_features,
        stem_features,
        contribution_matrix,
        channel_stems,
        channel_priorities,
        channel_measurements,
    ) = _build_autofoh_measurement_snapshot(rendered_channels, plans, sr, autofoh_config)

    if not channel_features or contribution_matrix is None:
        return {
            "enabled": False,
            "reason": "no_measurement_snapshot",
            "applied_actions": [],
            "detected_problems": [],
            "measurement_channels": channel_measurements,
        }

    lead_masking_config = detector_config.get("lead_masking", {})
    mud_config = detector_config.get("mud_excess", {})
    harshness_config = detector_config.get("harshness_excess", {})
    sibilance_config = detector_config.get("sibilance_excess", {})
    low_end_config = detector_config.get("low_end", {})

    lead_channels = [
        channel for channel, stems in channel_stems.items()
        if "LEAD" in stems or plans[channel].instrument == "lead_vocal"
    ]
    current_faders = {
        channel: float(plans[channel].fader_db)
        for channel in channel_features
    }

    analyzers = {
        "lead_masking": (
            bool(lead_masking_config.get("enabled", True)) and bool(lead_channels),
            LeadMaskingAnalyzer(
                masking_threshold_db=float(lead_masking_config.get("masking_threshold_db", 3.0)),
                culprit_share_threshold=float(lead_masking_config.get("min_culprit_contribution", 0.35)),
                persistence_required_cycles=1,
                lead_boost_db=float(lead_masking_config.get("lead_boost_db", 0.5)),
            ),
        ),
        "mud_excess": (
            bool(mud_config.get("enabled", True)),
            MudExcessDetector(
                threshold_db=float(mud_config.get("threshold_db", 2.5)),
                persistence_required_cycles=1,
                hysteresis_db=float(mud_config.get("hysteresis_db", 0.75)),
            ),
        ),
        "harshness_excess": (
            bool(harshness_config.get("enabled", True)),
            HarshnessExcessDetector(
                threshold_db=float(harshness_config.get("threshold_db", 2.5)),
                persistence_required_cycles=1,
                hysteresis_db=float(harshness_config.get("hysteresis_db", 0.75)),
            ),
        ),
        "sibilance_excess": (
            bool(sibilance_config.get("enabled", True)),
            SibilanceExcessDetector(
                threshold_db=float(sibilance_config.get("threshold_db", 2.5)),
                persistence_required_cycles=1,
                hysteresis_db=float(sibilance_config.get("hysteresis_db", 0.75)),
            ),
        ),
        "low_end": (
            bool(low_end_config.get("enabled", True)),
            LowEndAnalyzer(
                sub_threshold_db=float(low_end_config.get("sub_threshold_db", 4.0)),
                bass_threshold_db=float(low_end_config.get("bass_threshold_db", 3.0)),
                body_threshold_db=float(low_end_config.get("body_threshold_db", 2.5)),
                culprit_share_threshold=float(low_end_config.get("min_culprit_contribution", 0.35)),
                persistence_required_cycles=1,
                hysteresis_db=float(low_end_config.get("hysteresis_db", 0.75)),
            ),
        ),
    }

    recommendations: list[tuple[str, Any, list[TypedCorrectionAction]]] = []
    if analyzers["lead_masking"][0]:
        lead_result = analyzers["lead_masking"][1].analyze(
            channel_features=channel_features,
            channel_stems=channel_stems,
            stem_features=stem_features,
            contribution_matrix=contribution_matrix,
            lead_channel_ids=lead_channels,
            current_faders_db=current_faders,
            lead_priorities=channel_priorities,
            runtime_state=RuntimeState.PRE_SHOW_CHECK,
        )
        if lead_result.problem:
            recommendations.append(("lead_masking", lead_result.problem, lead_result.candidate_actions))

    recommendations.extend(
        _lead_foreground_balance_recommendations(
            plans,
            channel_features,
            stem_features,
            lead_channels,
        )
    )
    recommendations.extend(
        _lead_handoff_balance_recommendations(
            plans,
            channel_features,
            lead_channels,
        )
    )
    recommendations.extend(
        _lead_background_hierarchy_recommendations(
            plans,
            channel_features,
            lead_channels,
        )
    )

    master_features = stem_features.get("MASTER")
    if master_features is not None:
        if analyzers["low_end"][0]:
            low_end_result = analyzers["low_end"][1].analyze(
                master_features=master_features,
                contribution_matrix=contribution_matrix,
                channel_features=channel_features,
                channel_stems=channel_stems,
            )
            if low_end_result.problem:
                recommendations.append(("low_end", low_end_result.problem, low_end_result.candidate_actions))

        for label in ("mud_excess", "harshness_excess", "sibilance_excess"):
            enabled, detector = analyzers[label]
            if not enabled:
                continue
            recommendation = detector.observe(
                master_features=master_features,
                contribution_matrix=contribution_matrix,
                channel_features=channel_features,
                channel_stems=channel_stems,
            )
            if recommendation.problem:
                recommendations.append((label, recommendation.problem, recommendation.candidate_actions))

        recommendations.extend(
            _cymbal_buildup_recommendations(
                plans,
                channel_features,
                master_features,
            )
        )

    adapter = OfflineAutoFOHConsole(plans)
    safety_controller = AutoFOHSafetyController(adapter, config=safety_config)
    applied_actions = []
    detected_problems = []
    for label, problem, actions in recommendations:
        detected_problems.append({
            "label": label,
            "problem": _problem_to_dict(problem),
            "candidate_actions": [_typed_action_to_dict(action) for action in actions],
        })
        if not actions:
            continue
        action = actions[0]
        decision = safety_controller.execute(action, RuntimeState.PRE_SHOW_CHECK)
        action_report = {
            "label": label,
            "requested_action": _typed_action_to_dict(action),
            "applied_action": _typed_action_to_dict(decision.action),
            "sent": bool(decision.sent),
            "bounded": bool(decision.bounded),
            "allowed": bool(decision.allowed),
            "supported": bool(decision.supported),
            "message": decision.message,
            "problem": _problem_to_dict(problem),
        }
        applied_actions.append(action_report)
        if decision.sent:
            channel_id = getattr(decision.action, "channel_id", None)
            if channel_id in plans:
                plans[channel_id].autofoh_actions.append(action_report)

    master_indexes = {}
    if master_features is not None:
        master_indexes = {
            key: round(float(value), 3)
            for key, value in master_features.mix_indexes.as_dict().items()
        }

    return {
        "enabled": True,
        "measurement_mode": "autofoh_analyzers",
        "measurement_channels": channel_measurements,
        "master_indexes": master_indexes,
        "detected_problems": detected_problems,
        "applied_actions": applied_actions,
        "virtual_console_calls": adapter.calls,
        "notes": [
            "All additional offline correction moves in this pass come from measured AutoFOH detector outputs.",
            "Lead handoff and cymbal buildup decisions are derived from the same measured analyzer snapshot.",
            "Legacy codex heuristic correction layers are disabled in analyzer-only mode.",
        ],
    }


def apply_autofoh_analyzer_pass(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    autofoh_config: dict[str, Any] | None = None,
    *,
    max_rounds: int = DEFAULT_AUTOFOH_ANALYZER_ROUNDS,
    render_cache: OfflineRenderCache | None = None,
    analysis_preview_sec: float = ANALYZER_RENDER_PREVIEW_SEC,
) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    max_rounds = max(1, int(max_rounds))
    for round_index in range(1, max_rounds + 1):
        rendered_channels = {
            channel: render_channel_preview_cached(
                channel,
                plan,
                sr,
                preview_sec=analysis_preview_sec,
                render_cache=render_cache,
            )
            for channel, plan in plans.items()
            if not plan.muted
        }
        round_report = apply_autofoh_measurement_corrections(
            plans,
            rendered_channels,
            sr,
            autofoh_config=autofoh_config,
        )
        round_report["round"] = round_index
        rounds.append(round_report)
        if not any(action.get("sent") for action in round_report.get("applied_actions", [])):
            break

    if not rounds:
        return {
            "enabled": False,
            "reason": "no_rounds_executed",
            "rounds": [],
        }

    combined_detected = []
    combined_actions = []
    for round_report in rounds:
        for item in round_report.get("detected_problems", []):
            enriched = dict(item)
            enriched["round"] = round_report["round"]
            combined_detected.append(enriched)
        for item in round_report.get("applied_actions", []):
            enriched = dict(item)
            enriched["round"] = round_report["round"]
            combined_actions.append(enriched)

    final_report = dict(rounds[-1])
    final_report["measurement_mode"] = "autofoh_analyzers_iterative"
    final_report["round_count"] = len(rounds)
    final_report["max_rounds"] = max_rounds
    final_report["analysis_preview_sec"] = round(float(analysis_preview_sec), 2)
    final_report["rounds"] = rounds
    final_report["detected_problems"] = combined_detected
    final_report["applied_actions"] = combined_actions
    final_report["notes"] = list(final_report.get("notes", [])) + [
        "Offline analyzer mode uses bounded preview renders for fast measurement.",
        "Increase --autofoh-rounds for a slower deep convergence pass.",
    ]
    return final_report


def apply_vocal_bed_balance(
    plans: dict[int, ChannelPlan],
    desired_vocal_delta_db: float = 6.5,
    max_bed_attenuation_db: float = 3.0,
    protected_instruments: tuple[str, ...] = ("kick", "snare", "bass_guitar"),
) -> dict[str, Any]:
    """Lower the whole music bed when vocal cannot safely move far enough up."""
    vocal_plans = [
        (channel, plan)
        for channel, plan in plans.items()
        if plan.instrument == "lead_vocal" and not plan.muted
    ]
    if not vocal_plans:
        return {}

    vocal_channel, vocal_plan = max(vocal_plans, key=lambda item: _post_fader_lufs(item[1]))
    music_lufs = _music_bed_lufs(plans)
    vocal_lufs = _post_fader_lufs(vocal_plan)
    current_delta = vocal_lufs - music_lufs
    shortfall = desired_vocal_delta_db - current_delta
    attenuation = max(0.0, min(max_bed_attenuation_db, shortfall))

    adjusted_channels = []
    if attenuation >= 0.25:
        for channel, plan in plans.items():
            if (
                plan.instrument == "lead_vocal"
                or plan.muted
                or plan.instrument in protected_instruments
            ):
                continue
            plan.fader_db = float(np.clip(plan.fader_db - attenuation, -100.0, 10.0))
            adjusted_channels.append(channel)

    return {
        "vocal_channel": vocal_channel,
        "desired_vocal_delta_db": round(desired_vocal_delta_db, 2),
        "before_vocal_lufs": round(vocal_lufs, 2),
        "before_music_bed_lufs": round(music_lufs, 2),
        "before_delta_db": round(current_delta, 2),
        "bed_attenuation_db": round(attenuation, 2),
        "after_vocal_lufs": round(_post_fader_lufs(vocal_plan), 2),
        "after_music_bed_lufs": round(_music_bed_lufs(plans), 2),
        "after_delta_db": round(_post_fader_lufs(vocal_plan) - _music_bed_lufs(plans), 2),
        "protected_instruments": list(protected_instruments),
        "adjusted_channels": adjusted_channels,
    }


def codex_correction_actions(plans: dict[int, ChannelPlan]) -> list[AgentAction]:
    """Human-in-the-loop correction pass used when Codex is the mix agent."""
    actions: list[AgentAction] = []

    vocal_plans = [
        (channel, plan)
        for channel, plan in plans.items()
        if plan.instrument == "lead_vocal" and not plan.muted
    ]
    if len(vocal_plans) > 1:
        anchor_channel, anchor_plan = max(vocal_plans, key=lambda item: _post_fader_lufs(item[1]))
        anchor_lufs = _post_fader_lufs(anchor_plan)
        for channel, plan in vocal_plans:
            if channel == anchor_channel:
                continue
            target_lufs = anchor_lufs - 1.2
            shortfall = target_lufs - _post_fader_lufs(plan)
            if shortfall >= 0.5:
                adjustment = float(np.clip(shortfall, 0.0, 2.0))
                actions.append(AgentAction(
                    action_type="adjust_gain",
                    channel=channel,
                    parameters={"channel": channel, "adjustment_db": adjustment},
                    priority=1,
                    confidence=0.72,
                    reason=(
                        "Codex correction: keep additional lead vocals within about "
                        "1-1.5 dB of the loudest lead so verse/feature lines do not drop behind the band."
                    ),
                    source="codex",
                    risk="low",
                    expected_effect="More even lead-vocal handoff without making all vocal mics equally dominant.",
                    rollback_hint=f"Lower channel {channel} fader by {adjustment:.1f} dB.",
                ))

    for channel, plan in plans.items():
        if plan.instrument != "backing_vocal" or plan.muted:
            continue
        actions.append(AgentAction(
            action_type="reduce_gain",
            channel=channel,
            parameters={"channel": channel, "amount_db": -2.0},
            priority=2,
            confidence=0.74,
            reason=(
                "Codex correction: backing vocals are already compressed and read too forward, "
                "so lower the pair instead of adding more compression or presence."
            ),
            source="codex",
            risk="low",
            expected_effect="Back vocals sit behind the lead vocals with less apparent loudness.",
            rollback_hint=f"Raise channel {channel} fader by 2.0 dB.",
        ))

    return actions


def _set_compressor_plan(plan: ChannelPlan, threshold_db: float, ratio: float, attack_ms: float, release_ms: float):
    plan.comp_threshold_db = float(threshold_db)
    plan.comp_ratio = float(ratio)
    plan.comp_attack_ms = float(attack_ms)
    plan.comp_release_ms = float(release_ms)


def apply_codex_bleed_control(plans: dict[int, ChannelPlan]) -> dict[str, Any]:
    """Reduce cymbal bleed buildup and avoid re-compressing pre-compressed backing vocals."""
    changes: list[dict[str, Any]] = []

    def record(channel: int, plan: ChannelPlan, change: str, before: Any, after: Any, reason: str):
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "change": change,
            "before": before,
            "after": after,
            "reason": reason,
            "source": "codex",
        })

    def add_eq_cut(channel: int, plan: ChannelPlan, freq: float, gain: float, q: float, reason: str):
        before = list(plan.eq_bands)
        plan.eq_bands.append((freq, gain, q))
        record(channel, plan, "append_eq_band", before, list(plan.eq_bands), reason)

    for channel, plan in plans.items():
        if plan.muted:
            continue

        if plan.instrument == "backing_vocal":
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            _set_compressor_plan(plan, 0.0, 1.0, 10.0, 100.0)
            record(
                channel,
                plan,
                "disable_channel_compression",
                before_comp,
                {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio},
                "Backing vocals are already compressed in the source; extra compression makes them appear louder than the leads.",
            )
            before_lpf = plan.lpf
            plan.lpf = 9500.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Keep backing vocal air controlled after removing extra compression.")

        if plan.instrument in {"kick", "snare", "rack_tom", "floor_tom"}:
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            if plan.instrument == "kick":
                _set_compressor_plan(plan, -10.0, 1.4, 12.0, 120.0)
                lpf = 6800.0
            elif plan.instrument == "snare":
                _set_compressor_plan(plan, -11.0, 1.5, 12.0, 130.0)
                lpf = 7600.0
            else:
                _set_compressor_plan(plan, 0.0, 1.0, 12.0, 160.0)
                lpf = 5800.0
            record(
                channel,
                plan,
                "soften_or_disable_drum_compression",
                before_comp,
                {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio, "attack_ms": plan.comp_attack_ms, "release_ms": plan.comp_release_ms},
                "Close drum mics contain strong cymbal bleed; compression was pulling that bleed forward.",
            )
            before_lpf = plan.lpf
            plan.lpf = lpf
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Low-pass close drum mics so cymbal spill does not stack across the kit.")
            add_eq_cut(channel, plan, 6500.0, -2.0, 1.1, "Presence cut targets cymbal bleed in close drum mics.")
            add_eq_cut(channel, plan, 9500.0, -3.0, 0.9, "Air-band cut reduces repeated cymbal spill across close drum mics.")

        if plan.instrument in {"hi_hat", "ride"}:
            before_fader = plan.fader_db
            plan.fader_db = float(np.clip(plan.fader_db - 2.5, -100.0, 10.0))
            record(channel, plan, "adjust_fader_db", round(before_fader, 2), round(plan.fader_db, 2), "Direct cymbal spot mics were adding to already heavy cymbal bleed.")
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            _set_compressor_plan(plan, 0.0, 1.0, 20.0, 180.0)
            record(channel, plan, "disable_channel_compression", before_comp, {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio}, "Do not compress cymbal spot mics when the kit already has cymbal spill.")
            before_lpf = plan.lpf
            plan.lpf = 10500.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Keep cymbal spot tone present but less splashy.")

        if plan.instrument == "overhead":
            before_fader = plan.fader_db
            plan.fader_db = float(np.clip(plan.fader_db - 1.2, -100.0, 10.0))
            record(channel, plan, "adjust_fader_db", round(before_fader, 2), round(plan.fader_db, 2), "Overheads are the main cymbal picture, but the rest of the kit also contains cymbal bleed.")
            before_lpf = plan.lpf
            plan.lpf = 9000.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Softer overhead top end for a less cymbal-heavy drum image.")
            add_eq_cut(channel, plan, 6500.0, -1.5, 1.0, "Presence cut reduces harsh cymbal build-up.")

        if plan.instrument == "room":
            before_fader = plan.fader_db
            plan.fader_db = float(np.clip(plan.fader_db - 2.0, -100.0, 10.0))
            record(channel, plan, "adjust_fader_db", round(before_fader, 2), round(plan.fader_db, 2), "Room mics are useful for space but add broad cymbal wash in this multitrack.")
            before_comp = {
                "threshold_db": plan.comp_threshold_db,
                "ratio": plan.comp_ratio,
                "attack_ms": plan.comp_attack_ms,
                "release_ms": plan.comp_release_ms,
            }
            _set_compressor_plan(plan, 0.0, 1.0, 20.0, 220.0)
            record(channel, plan, "disable_channel_compression", before_comp, {"threshold_db": plan.comp_threshold_db, "ratio": plan.comp_ratio}, "Avoid lifting cymbal decay from room microphones.")
            before_lpf = plan.lpf
            plan.lpf = 6500.0
            record(channel, plan, "set_lpf_hz", before_lpf, plan.lpf, "Keep room size while reducing cymbal splash.")

    return {
        "enabled": True,
        "changes": changes,
        "notes": [
            "Cymbal bleed is handled by reducing high-frequency buildup at every drum capture point, not only overheads.",
            "Backing vocals are treated as pre-compressed sources: lower level, no extra compressor.",
        ],
    }


def _frame_rms_db(x: np.ndarray, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    mono = mono_sum(x) if x.ndim > 1 else x
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    starts = np.arange(0, max(1, len(mono) - frame + 1), hop, dtype=np.int64)
    values = []
    for start in starts:
        block = mono[start:start + frame]
        values.append(amp_to_db(float(np.sqrt(np.mean(np.square(block))) + 1e-12)))
    return starts.astype(np.float32), np.asarray(values, dtype=np.float32)


def _rolling_percentile(
    values: np.ndarray,
    radius: int,
    percentile: float = 60.0,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        window = values[start:end]
        if active_mask is not None:
            active_window = window[active_mask[start:end]]
            if len(active_window) >= max(3, len(window) // 4):
                window = active_window
        out[index] = float(np.percentile(window, percentile)) if len(window) else float(values[index])
    return out


def _frame_gain_to_samples(gain_db: np.ndarray, starts: np.ndarray, frame: int, total_len: int) -> np.ndarray:
    if total_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(gain_db) == 0:
        return np.ones(total_len, dtype=np.float32)
    sample_points = np.clip(starts + frame // 2, 0, total_len - 1)
    full_points = np.concatenate(([0.0], sample_points.astype(np.float32), [float(total_len - 1)]))
    full_gain = np.concatenate(([gain_db[0]], gain_db, [gain_db[-1]])).astype(np.float32)
    envelope_db = np.interp(np.arange(total_len, dtype=np.float32), full_points, full_gain).astype(np.float32)
    return np.power(10.0, envelope_db / 20.0, dtype=np.float32)


def _reference_dynamics_tightness(reference_context: ReferenceMixContext | None) -> float:
    if reference_context is None:
        return 0.0
    style = reference_context.style_profile
    style_tightness = float(np.clip((15.5 - float(style.dynamic_range)) / 8.0, 0.0, 1.0))
    section_lufs = [float(section.get("lufs")) for section in reference_context.sections if "lufs" in section]
    if len(section_lufs) >= 2:
        section_span = max(section_lufs) - min(section_lufs)
        macro_tightness = float(np.clip((5.5 - section_span) / 4.5, 0.0, 1.0))
        return float(np.clip(style_tightness * 0.75 + macro_tightness * 0.25, 0.0, 1.0))
    return style_tightness


def _reference_fx_space_lift(reference_context: ReferenceMixContext | None) -> float:
    if reference_context is None:
        return 0.0
    style = reference_context.style_profile
    width = float(np.clip(style.stereo_width, 0.0, 1.0))
    tightness = float(np.clip((8.0 - float(style.dynamic_range)) / 8.0, 0.0, 1.0))
    darkness = float(np.clip(
        (float(style.spectral_balance.get("mid", -11.5)) - float(style.spectral_balance.get("presence", -19.5))) / 10.0,
        0.0,
        1.4,
    ))
    wetness = 0.24 + width * 0.82 + darkness * 0.34 + tightness * 0.28
    return float(np.clip(wetness, 0.18, 1.55))


def _channel_family_for_dynamics(plan: ChannelPlan) -> str | None:
    if plan.instrument == "lead_vocal":
        return "lead_vocals"
    if plan.instrument == "backing_vocal":
        return "backing_vocals"
    if plan.instrument in DRUM_INSTRUMENTS or plan.instrument in {"overhead", "room"}:
        return "drums"
    if plan.instrument in {"bass", "bass_di", "bass_mic", "bass_guitar", "synth_bass"}:
        return "bass"
    if plan.instrument in {
        "electric_guitar",
        "guitar",
        "acoustic_guitar",
        "keys",
        "piano",
        "organ",
        "synth",
        "pad",
        "lead_guitar",
        "rhythm_guitar",
        "accordion",
        "playback",
        "tracks",
        "music",
    }:
        return "music"
    return None


def _dynamics_profile_for_family(family: str, tightness: float) -> dict[str, float]:
    if family == "lead_vocals":
        return {
            "frame_sec": 0.30,
            "hop_sec": 0.08,
            "window_sec": 8.0,
            "active_margin_db": 16.0,
            "percentile": 56.0,
            "cut_strength": 0.54 + tightness * 0.14,
            "boost_strength": 0.18 + tightness * 0.07,
            "peak_shave_db": 0.8 + tightness * 0.7,
            "max_cut_db": 2.8 + tightness * 1.0,
            "max_boost_db": 0.9 + tightness * 0.45,
            "attack_ms": 140.0,
            "release_ms": 980.0,
        }
    if family == "backing_vocals":
        return {
            "frame_sec": 0.34,
            "hop_sec": 0.10,
            "window_sec": 8.5,
            "active_margin_db": 16.0,
            "percentile": 58.0,
            "cut_strength": 0.46 + tightness * 0.12,
            "boost_strength": 0.12 + tightness * 0.05,
            "peak_shave_db": 0.55 + tightness * 0.45,
            "max_cut_db": 2.2 + tightness * 0.8,
            "max_boost_db": 0.7 + tightness * 0.3,
            "attack_ms": 180.0,
            "release_ms": 1200.0,
        }
    if family == "drums":
        return {
            "frame_sec": 0.26,
            "hop_sec": 0.07,
            "window_sec": 6.5,
            "active_margin_db": 18.0,
            "percentile": 54.0,
            "cut_strength": 0.28 + tightness * 0.10,
            "boost_strength": 0.07 + tightness * 0.03,
            "peak_shave_db": 0.4 + tightness * 0.35,
            "max_cut_db": 1.8 + tightness * 0.6,
            "max_boost_db": 0.45 + tightness * 0.2,
            "attack_ms": 90.0,
            "release_ms": 720.0,
        }
    if family == "bass":
        return {
            "frame_sec": 0.32,
            "hop_sec": 0.09,
            "window_sec": 7.0,
            "active_margin_db": 16.0,
            "percentile": 55.0,
            "cut_strength": 0.22 + tightness * 0.08,
            "boost_strength": 0.08 + tightness * 0.03,
            "peak_shave_db": 0.3 + tightness * 0.2,
            "max_cut_db": 1.5 + tightness * 0.4,
            "max_boost_db": 0.35 + tightness * 0.15,
            "attack_ms": 120.0,
            "release_ms": 900.0,
        }
    return {
        "frame_sec": 0.36,
        "hop_sec": 0.10,
        "window_sec": 8.5,
        "active_margin_db": 16.0,
        "percentile": 57.0,
        "cut_strength": 0.3 + tightness * 0.08,
        "boost_strength": 0.09 + tightness * 0.03,
        "peak_shave_db": 0.35 + tightness * 0.2,
        "max_cut_db": 1.7 + tightness * 0.5,
        "max_boost_db": 0.45 + tightness * 0.18,
        "attack_ms": 160.0,
        "release_ms": 980.0,
    }


def _smooth_gain_db(target_db: np.ndarray, sr: int, hop: int, attack_ms: float = 180.0, release_ms: float = 900.0) -> np.ndarray:
    attack = math.exp(-hop / max(1.0, attack_ms * 0.001 * sr))
    release = math.exp(-hop / max(1.0, release_ms * 0.001 * sr))
    smoothed = np.zeros_like(target_db, dtype=np.float32)
    last = 0.0
    for i, value in enumerate(target_db):
        coeff = attack if value < last else release
        last = coeff * last + (1.0 - coeff) * float(value)
        smoothed[i] = last
    return smoothed


def apply_dynamic_vocal_priority(
    rendered: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    max_duck_db: float = 3.5,
    duck_drum_instruments: bool = True,
) -> dict[str, Any]:
    """Apply chorus-aware DCA ducking of non-vocal channels while vocal is active.

    If ducking is disabled for drums, kick/snare/toms/cymbal channels stay untouched.
    """
    vocal_channels = [
        channel for channel, plan in plans.items()
        if plan.instrument == "lead_vocal" and channel in rendered and not plan.muted
    ]
    if not vocal_channels:
        return {}

    vocal = sum((rendered[channel] for channel in vocal_channels), np.zeros_like(next(iter(rendered.values()))))
    bed = sum(
        (audio for channel, audio in rendered.items() if channel not in vocal_channels and not plans[channel].muted),
        np.zeros_like(vocal),
    )

    frame = int(0.35 * sr)
    hop = int(0.08 * sr)
    starts, vocal_db = _frame_rms_db(vocal, frame, hop)
    _, bed_db = _frame_rms_db(bed, frame, hop)
    active = vocal_db > -36.0
    if not np.any(active):
        return {}

    active_bed = bed_db[active]
    bed_mid = float(np.percentile(active_bed, 55))
    bed_loud = float(np.percentile(active_bed, 88))
    density_span = max(1.0, bed_loud - bed_mid)
    density = np.clip((bed_db - bed_mid) / density_span, 0.0, 1.0)

    # Stronger ducking is only requested when the arrangement gets dense and
    # the vocal-to-bed ratio is not generous enough for a live-style lead.
    vocal_margin = vocal_db - bed_db
    margin_shortfall = np.clip((4.0 - vocal_margin) / 8.0, 0.0, 1.0)
    duck_db = -max_duck_db * density * margin_shortfall
    duck_db = np.where(active, duck_db, 0.0).astype(np.float32)
    duck_db = _smooth_gain_db(duck_db, sr, hop)

    sample_points = np.clip(starts + frame // 2, 0, len(vocal) - 1)
    full_points = np.concatenate(([0.0], sample_points, [float(len(vocal) - 1)]))
    full_duck = np.concatenate(([duck_db[0]], duck_db, [duck_db[-1]]))
    envelope_db = np.interp(np.arange(len(vocal)), full_points, full_duck).astype(np.float32)
    gain = (10.0 ** (envelope_db / 20.0)).astype(np.float32)

    adjusted_channels = []
    for channel, audio in rendered.items():
        if channel in vocal_channels or plans[channel].muted:
            continue
        if not duck_drum_instruments and plans[channel].instrument in DRUM_INSTRUMENTS:
            continue
        rendered[channel] = (audio * gain[:, None]).astype(np.float32)
        adjusted_channels.append(channel)

    ducked = duck_db < -0.25
    return {
        "vocal_channels": vocal_channels,
        "adjusted_channels": adjusted_channels,
        "max_duck_db": round(abs(float(np.min(duck_db))), 2),
        "mean_duck_when_active_db": round(abs(float(np.mean(duck_db[active]))), 2),
        "ducked_sec": round(float(np.sum(ducked) * hop / sr), 2),
        "active_vocal_sec": round(float(np.sum(active) * hop / sr), 2),
        "bed_mid_db": round(bed_mid, 2),
        "bed_loud_db": round(bed_loud, 2),
    }


def apply_reference_dynamics_ride(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    reference_context: ReferenceMixContext | None,
) -> dict[str, Any]:
    if reference_context is None:
        return {"enabled": False, "reason": "no_reference_supplied"}
    if not rendered_channels:
        return {"enabled": False, "reason": "no_rendered_channels"}

    tightness = _reference_dynamics_tightness(reference_context)
    if tightness <= 0.02:
        return {
            "enabled": False,
            "reason": "reference_is_not_tight_enough_for_extra_rides",
            "tightness": round(tightness, 3),
        }

    first_audio = next(iter(rendered_channels.values()))
    group_channels: dict[str, list[int]] = {}
    for channel, audio in rendered_channels.items():
        if len(audio) == 0:
            continue
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        family = _channel_family_for_dynamics(plan)
        if family is None:
            continue
        group_channels.setdefault(family, []).append(channel)

    groups_report = []
    adjusted_channel_count = 0
    for family, channels in sorted(group_channels.items()):
        profile = _dynamics_profile_for_family(family, tightness)
        group_audio = sum((rendered_channels[channel] for channel in channels), np.zeros_like(first_audio))
        frame = max(1024, int(profile["frame_sec"] * sr))
        hop = max(256, int(profile["hop_sec"] * sr))
        starts, rms_db = _frame_rms_db(group_audio, frame, hop)
        if len(rms_db) < 4:
            continue

        p90 = float(np.percentile(rms_db, 90))
        p60 = float(np.percentile(rms_db, 60))
        active_threshold = max(-54.0, min(p60 - 5.0, p90 - profile["active_margin_db"]))
        active = rms_db > active_threshold
        if int(np.sum(active)) < max(6, len(rms_db) // 10):
            continue

        radius = max(2, int(profile["window_sec"] * sr / max(hop, 1) / 2.0))
        local_target = _rolling_percentile(rms_db, radius, percentile=profile["percentile"], active_mask=active)
        diff_db = local_target - rms_db
        gain_target = np.where(diff_db < 0.0, diff_db * profile["cut_strength"], diff_db * profile["boost_strength"]).astype(np.float32)

        loud_anchor = float(np.percentile(rms_db[active], 82))
        loud_excess = np.clip((rms_db - loud_anchor) / max(1.4, 3.0 - tightness * 0.7), 0.0, 1.0).astype(np.float32)
        gain_target -= loud_excess * profile["peak_shave_db"]

        gain_target = np.where(active, gain_target, 0.0).astype(np.float32)
        gain_target = np.clip(gain_target, -profile["max_cut_db"], profile["max_boost_db"]).astype(np.float32)
        gain_smoothed = _smooth_gain_db(
            gain_target,
            sr,
            hop,
            attack_ms=profile["attack_ms"],
            release_ms=profile["release_ms"],
        )
        gain = _frame_gain_to_samples(gain_smoothed, starts, frame, len(group_audio))
        if len(gain) == 0:
            continue

        for channel in channels:
            rendered_channels[channel] = (rendered_channels[channel] * gain[:, None]).astype(np.float32)
        adjusted_channel_count += len(channels)

        active_gain = gain_smoothed[active]
        cut_frames = gain_smoothed < -0.2
        boost_frames = gain_smoothed > 0.2
        groups_report.append({
            "family": family,
            "channels": channels,
            "files": [plans[channel].path.name for channel in channels if channel in plans],
            "active_threshold_db": round(active_threshold, 2),
            "active_sec": round(float(np.sum(active) * hop / sr), 2),
            "max_cut_db": round(abs(float(np.min(gain_smoothed))), 2),
            "max_boost_db": round(float(np.max(gain_smoothed)), 2),
            "mean_gain_when_active_db": round(float(np.mean(active_gain)) if len(active_gain) else 0.0, 2),
            "cut_sec": round(float(np.sum(cut_frames) * hop / sr), 2),
            "boost_sec": round(float(np.sum(boost_frames) * hop / sr), 2),
            "profile": {
                "cut_strength": round(profile["cut_strength"], 3),
                "boost_strength": round(profile["boost_strength"], 3),
                "peak_shave_db": round(profile["peak_shave_db"], 3),
                "window_sec": round(profile["window_sec"], 2),
            },
        })

    if not groups_report:
        return {
            "enabled": False,
            "reason": "no_groups_adjusted",
            "tightness": round(tightness, 3),
        }

    return {
        "enabled": True,
        "tightness": round(tightness, 3),
        "adjusted_channel_count": adjusted_channel_count,
        "groups": groups_report,
        "notes": [
            "This pass rides stem families toward a tighter local loudness window derived from the reference style.",
            "Lead vocals react fastest, drums are shaved more gently, and quiet gaps are not lifted.",
        ],
    }


def _priority_for_instrument(instrument: str) -> int:
    """Lower number means the channel is more important in mirror EQ."""
    if instrument == "lead_vocal":
        return 1
    if instrument == "kick":
        return 2
    if instrument in {"snare", "bass_guitar"}:
        return 3
    if instrument in {"electric_guitar", "accordion", "playback", "backing_vocal"}:
        return 4
    if instrument in {"overhead", "room", "rack_tom", "floor_tom", "hi_hat", "ride", "percussion"}:
        return 5
    return 4


def _analysis_block(x: np.ndarray, sr: int, window_sec: float = 18.0) -> np.ndarray:
    window = min(len(x), max(1024, int(window_sec * sr)))
    if len(x) <= window:
        return x
    hop = max(512, window // 3)
    best_start = 0
    best_energy = -1.0
    for start in range(0, len(x) - window, hop):
        energy = float(np.mean(np.square(x[start:start + window])))
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return x[best_start:best_start + window]


def _event_activity_ranges(x: np.ndarray, sr: int, instrument: str | None) -> dict[str, Any] | None:
    config = _event_metric_config(instrument or "")
    if not config or len(x) == 0:
        return None

    detect = highpass(x, sr, config["detect_hpf_hz"])
    detect = lowpass(detect, sr, config["detect_lpf_hz"])
    frame = max(256, int(config["frame_ms"] * sr / 1000.0))
    hop = max(64, int(config["hop_ms"] * sr / 1000.0))
    starts, rms_db = _frame_rms_db(detect, frame, hop)
    if len(rms_db) == 0:
        return {
            "config": config,
            "frame": frame,
            "hop": hop,
            "ranges": [],
            "threshold_db": None,
            "active_samples": 0,
        }

    peak_percentile = float(config.get("peak_percentile", 100.0))
    if len(detect):
        if peak_percentile >= 100.0:
            detect_peak = float(np.max(np.abs(detect)))
        else:
            detect_peak = float(np.percentile(np.abs(detect), peak_percentile))
    else:
        detect_peak = 0.0
    detect_peak_db = amp_to_db(detect_peak)
    noise_floor_db = float(np.percentile(rms_db, 50))
    threshold_db = max(
        float(np.percentile(rms_db, config["percentile"])),
        detect_peak_db - config["peak_offset_db"],
        noise_floor_db + config["floor_margin_db"],
        config["min_threshold_db"],
    )
    active_idx = np.flatnonzero(rms_db >= threshold_db)
    if len(active_idx) == 0:
        return {
            "config": config,
            "frame": frame,
            "hop": hop,
            "ranges": [],
            "threshold_db": round(threshold_db, 2),
            "active_samples": 0,
        }

    pad = int(config["pad_ms"] * sr / 1000.0)
    ranges = []
    for idx in active_idx:
        start = max(0, int(starts[idx]) - pad)
        end = min(len(x), int(starts[idx]) + frame + pad)
        if end > start:
            ranges.append((start, end))
    merged = _merge_ranges(ranges, gap=pad // 2)
    active_samples = sum(end - start for start, end in merged)
    return {
        "config": config,
        "frame": frame,
        "hop": hop,
        "ranges": merged,
        "threshold_db": round(threshold_db, 2),
        "active_samples": active_samples,
    }


def _ranges_to_mask(length: int, ranges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(max(0, int(length)), dtype=bool)
    for start, end in ranges:
        mask[max(0, int(start)):min(len(mask), int(end))] = True
    return mask


def _rms_db_for_samples(samples: np.ndarray) -> float | None:
    if len(samples) == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-12)
    return amp_to_db(rms)


def _activity_bleed_metrics(x: np.ndarray, sr: int, activity: dict[str, Any] | None) -> dict[str, Any]:
    if not activity or not activity.get("ranges") or len(x) == 0:
        return {
            "analysis_event_rms_db": None,
            "analysis_bleed_rms_db": None,
            "analysis_event_to_bleed_db": None,
            "analysis_bleed_ratio": None,
            "analysis_bleed_dominant": False,
        }

    mask = _ranges_to_mask(len(x), list(activity.get("ranges") or []))
    if not np.any(mask):
        return {
            "analysis_event_rms_db": None,
            "analysis_bleed_rms_db": None,
            "analysis_event_to_bleed_db": None,
            "analysis_bleed_ratio": None,
            "analysis_bleed_dominant": False,
        }

    active = x[mask]
    inactive = x[~mask]
    event_rms_db = _rms_db_for_samples(active)
    bleed_rms_db = _rms_db_for_samples(inactive)
    if event_rms_db is None or bleed_rms_db is None:
        event_to_bleed_db = None
        bleed_ratio = None
        bleed_dominant = False
    else:
        event_to_bleed_db = float(event_rms_db - bleed_rms_db)
        event_power = db_to_amp(event_rms_db) ** 2
        bleed_power = db_to_amp(bleed_rms_db) ** 2
        bleed_ratio = float(bleed_power / max(event_power + bleed_power, 1e-12))
        active_ratio = float(activity.get("active_samples", 0)) / max(1, len(x))
        bleed_dominant = bool(event_to_bleed_db < 8.0 or active_ratio > 0.35 or bleed_ratio > 0.2)

    return {
        "analysis_event_rms_db": round(event_rms_db, 2) if event_rms_db is not None else None,
        "analysis_bleed_rms_db": round(bleed_rms_db, 2) if bleed_rms_db is not None else None,
        "analysis_event_to_bleed_db": round(event_to_bleed_db, 2) if event_to_bleed_db is not None else None,
        "analysis_bleed_ratio": round(bleed_ratio, 4) if bleed_ratio is not None else None,
        "analysis_bleed_dominant": bleed_dominant,
    }


def _analysis_signal_for_metrics(x: np.ndarray, sr: int, instrument: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    activity = _event_activity_ranges(x, sr, instrument)
    if not activity:
        block = _analysis_block(x, sr)
        return block, {
            "analysis_mode": "windowed_full_track",
            "analysis_active_sec": round(len(block) / sr, 3) if sr else 0.0,
            "analysis_active_ratio": round(len(block) / max(1, len(x)), 4),
            "analysis_threshold_db": None,
            **_activity_bleed_metrics(x, sr, None),
        }

    merged = activity["ranges"]
    threshold_db = activity["threshold_db"]
    frame = activity["frame"]
    if not merged:
        block = _analysis_block(x, sr)
        return block, {
            "analysis_mode": "windowed_full_track_fallback",
            "analysis_active_sec": round(len(block) / sr, 3) if sr else 0.0,
            "analysis_active_ratio": round(len(block) / max(1, len(x)), 4),
            "analysis_threshold_db": round(threshold_db, 2),
            **_activity_bleed_metrics(x, sr, activity),
        }
    block = np.concatenate([x[start:end] for start, end in merged], axis=0) if merged else _analysis_block(x, sr)
    if len(block) < max(512, frame // 2):
        block = _analysis_block(x, sr)
        mode = "windowed_full_track_fallback"
    else:
        mode = "event_based"
    active_samples = activity["active_samples"]
    return block, {
        "analysis_mode": mode,
        "analysis_active_sec": round(active_samples / sr, 3) if sr else 0.0,
        "analysis_active_ratio": round(active_samples / max(1, len(x)), 4),
        "analysis_threshold_db": round(threshold_db, 2),
        **_activity_bleed_metrics(x, sr, activity),
    }


def compute_bleed_aware_trim(
    instrument: str,
    target_rms_db: float,
    metrics: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Compute input trim without letting bleed-heavy channels demand unsafe boosts."""
    raw_trim_db = float(target_rms_db - float(metrics.get("rms_db", -100.0)))
    max_boost_db = 12.0
    reasons: list[str] = []
    mode = str(metrics.get("analysis_mode") or "")
    active_ratio = float(metrics.get("analysis_active_ratio") or 0.0)
    event_to_bleed_db = metrics.get("analysis_event_to_bleed_db")
    bleed_ratio = metrics.get("analysis_bleed_ratio")
    bleed_dominant = bool(metrics.get("analysis_bleed_dominant"))

    if instrument in {"overhead", "room"}:
        max_boost_db = 6.0
        reasons.append("ambient mic: do not convert low-level kit bleed into large input gain")
    elif instrument in {"rack_tom", "floor_tom"}:
        if mode != "event_based":
            max_boost_db = 4.0
            reasons.append("tom detector did not find reliable primary hits")
        elif bleed_dominant:
            max_boost_db = 6.0
            reasons.append("tom event windows are not sufficiently separated from bleed")
        else:
            max_boost_db = 10.0
            reasons.append("tom primary hits are separated from bleed")
    elif instrument == "snare":
        if mode != "event_based" or bleed_dominant:
            max_boost_db = 7.0
            reasons.append("snare analysis is bleed-prone, cap corrective trim")
        else:
            max_boost_db = 10.0
            reasons.append("snare primary hits are separated from bleed")
    elif instrument in {"kick", "hi_hat", "ride", "percussion"}:
        if mode == "event_based" and not bleed_dominant:
            max_boost_db = 9.0
            reasons.append("percussive primary events are separated from bleed")
        else:
            max_boost_db = 6.0
            reasons.append("percussive channel is bleed-prone, cap corrective trim")
    elif instrument in {"lead_vocal", "backing_vocal"} and bleed_dominant:
        max_boost_db = 8.0
        reasons.append("vocal phrase analysis is close to bleed floor")

    trim_db = float(np.clip(raw_trim_db, -18.0, max_boost_db))
    return trim_db, {
        "raw_trim_db": round(raw_trim_db, 2),
        "applied_trim_db": round(trim_db, 2),
        "max_boost_db": round(max_boost_db, 2),
        "limited_by_bleed": bool(trim_db < raw_trim_db - 1e-6),
        "analysis_mode": mode,
        "active_ratio": round(active_ratio, 4),
        "event_to_bleed_db": round(float(event_to_bleed_db), 2) if event_to_bleed_db is not None else None,
        "bleed_ratio": round(float(bleed_ratio), 4) if bleed_ratio is not None else None,
        "bleed_dominant": bleed_dominant,
        "reasons": reasons,
    }


def _event_expander_profile(instrument: str, metrics: dict[str, Any]) -> dict[str, float] | None:
    if metrics.get("analysis_mode") != "event_based":
        return None

    active_ratio = float(metrics.get("analysis_active_ratio") or 0.0)
    threshold_db = metrics.get("analysis_threshold_db")
    if threshold_db is None:
        return None

    if instrument == "lead_vocal":
        return {
            "range_db": 5.8 if active_ratio < 0.3 else 4.8,
            "open_ms": 22.0,
            "close_ms": 200.0,
            "hold_ms": 240.0,
            "threshold_db": float(threshold_db) - 1.5,
        }
    if instrument == "backing_vocal":
        return {
            "range_db": 5.0 if active_ratio < 0.35 else 4.2,
            "open_ms": 22.0,
            "close_ms": 180.0,
            "hold_ms": 210.0,
            "threshold_db": float(threshold_db) - 0.8,
        }
    if instrument in {"rack_tom", "floor_tom"}:
        return {
            "range_db": 12.0 if active_ratio < 0.08 else 10.0,
            "open_ms": 10.0,
            "close_ms": 110.0,
            "hold_ms": 130.0,
            "threshold_db": float(threshold_db),
        }
    if instrument == "kick":
        return {
            "range_db": 7.5 if active_ratio < 0.2 else 6.0,
            "open_ms": 8.0,
            "close_ms": 115.0,
            "hold_ms": 130.0,
            "threshold_db": float(threshold_db),
        }
    if instrument == "snare":
        return {
            "range_db": 8.5 if active_ratio < 0.22 else 7.0,
            "open_ms": 8.0,
            "close_ms": 120.0,
            "hold_ms": 145.0,
            "threshold_db": float(threshold_db),
        }
    if instrument in {"hi_hat", "ride", "percussion"}:
        return {
            "range_db": 5.0 if active_ratio < 0.45 else 4.0,
            "open_ms": 14.0,
            "close_ms": 160.0,
            "hold_ms": 180.0,
            "threshold_db": float(threshold_db),
        }
    return None


def apply_event_based_dynamics(plans: dict[int, ChannelPlan]) -> dict[str, Any]:
    """Configure gentle event-based expanders so bleed does not drive dynamics."""
    changes: list[dict[str, Any]] = []
    for channel, plan in plans.items():
        if plan.muted:
            continue
        profile = _event_expander_profile(plan.instrument, plan.metrics)
        if not profile:
            continue
        plan.expander_enabled = True
        plan.expander_range_db = float(profile["range_db"])
        plan.expander_open_ms = float(profile["open_ms"])
        plan.expander_close_ms = float(profile["close_ms"])
        plan.expander_hold_ms = float(profile["hold_ms"])
        plan.expander_threshold_db = float(profile["threshold_db"])
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "analysis_active_ratio": round(float(plan.metrics.get("analysis_active_ratio") or 0.0), 4),
            "analysis_threshold_db": plan.expander_threshold_db,
            "range_db": round(plan.expander_range_db, 2),
            "open_ms": round(plan.expander_open_ms, 2),
            "close_ms": round(plan.expander_close_ms, 2),
            "hold_ms": round(plan.expander_hold_ms, 2),
            "reason": "Event-based activity windows drive a soft expander so bleed between hits or phrases does not keep the channel artificially open.",
        })

    return {
        "enabled": bool(changes),
        "changes": changes,
        "notes": [
            "Lead and backing vocals get a gentle downward expander between phrases.",
            "Rack and floor toms get a stronger event-based expander between hits.",
            "Kick, snare, and cymbal spot mics also use event-based expansion when bleed would otherwise keep them open.",
            "Ambient microphones such as overheads and rooms stay ungated.",
        ],
    }


def _band_energy_for_rendered(audio: np.ndarray, sr: int) -> dict[str, float]:
    mono = mono_sum(audio)
    block = _analysis_block(mono, sr)
    if len(block) < 1024:
        return {}
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    bands = {
        "sub": (20.0, 60.0),
        "bass": (60.0, 250.0),
        "low_mid": (250.0, 500.0),
        "mid": (500.0, 2000.0),
        "high_mid": (2000.0, 4000.0),
        "high": (4000.0, 8000.0),
        "air": (8000.0, 14000.0),
    }
    out = {}
    for name, (lo, hi) in bands.items():
        idx = (freqs >= lo) & (freqs < hi)
        if not np.any(idx):
            out[name] = -100.0
            continue
        band_rms = float(np.sqrt(np.sum(np.square(spec[idx]))) / len(windowed))
        out[name] = amp_to_db(band_rms)
    return out


def _cross_band_for_frequency(freq: float) -> str:
    centers = CrossAdaptiveEQ.BAND_CENTERS
    return min(centers.keys(), key=lambda name: abs(centers[name] - freq))


def apply_cross_adaptive_eq(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    render_cache: OfflineRenderCache | None = None,
    analysis_preview_sec: float = ANALYZER_RENDER_PREVIEW_SEC,
) -> dict[str, Any]:
    """Run priority-based CrossAdaptiveEQ and add conservative channel EQ moves."""
    preview = {
        channel: render_channel_preview_cached(
            channel,
            plan,
            sr,
            preview_sec=analysis_preview_sec,
            render_cache=render_cache,
        )
        for channel, plan in plans.items()
        if not plan.muted
    }
    channel_band_energy = {
        channel: _band_energy_for_rendered(audio, sr)
        for channel, audio in preview.items()
    }
    channel_priorities = {
        channel: _priority_for_instrument(plan.instrument)
        for channel, plan in plans.items()
        if channel in preview
    }

    processor = CrossAdaptiveEQ(
        min_band_level_db=-82.0,
        overlap_tolerance_db=8.0,
        max_cut_db=-2.5,
        max_boost_db=1.2,
    )
    raw_adjustments = processor.calculate_corrections(channel_band_energy, channel_priorities)

    aggregated: dict[tuple[int, float], dict[str, Any]] = {}
    skipped = 0
    for adj in raw_adjustments:
        channel = int(adj.channel_id)
        plan = plans.get(channel)
        if not plan:
            skipped += 1
            continue
        band = _cross_band_for_frequency(float(adj.frequency_hz))
        priority = channel_priorities.get(channel, 3)
        gain = float(adj.gain_db)

        if band == "air":
            skipped += 1
            continue
        if plan.instrument == "kick" and gain < 0.0 and band in {"sub", "bass", "low_mid"}:
            skipped += 1
            continue
        if gain < 0.0 and priority == 1:
            skipped += 1
            continue
        if gain > 0.0 and priority != 1:
            skipped += 1
            continue
        if gain > 0.0 and band in {"sub", "bass", "low_mid"}:
            skipped += 1
            continue

        if band in {"mid", "high_mid", "high"}:
            scale = 0.36 if gain < 0.0 else 0.32
        elif band in {"low_mid", "bass"}:
            scale = 0.24 if gain < 0.0 else 0.20
        else:
            scale = 0.20
        if plan.instrument == "bass_guitar" and band in {"sub", "bass"} and gain < 0.0:
            scale = 0.36

        key = (channel, float(adj.frequency_hz))
        existing = aggregated.setdefault(key, {
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "priority": priority,
            "frequency_hz": float(adj.frequency_hz),
            "band": band,
            "gain_db": 0.0,
            "q": float(adj.q_factor),
            "source_count": 0,
        })
        existing["gain_db"] += gain * scale
        existing["source_count"] += 1

    applied = []
    by_channel: dict[int, list[dict[str, Any]]] = {}
    for item in aggregated.values():
        channel = int(item["channel"])
        priority = int(item["priority"])
        gain = float(item["gain_db"])
        if gain < 0.0:
            if plans[channel].instrument == "bass_guitar" and item["band"] in {"sub", "bass"}:
                lower = -1.8
            else:
                lower = -2.5 if priority >= 4 else (-2.0 if priority == 3 else -1.0)
            gain = max(lower, gain)
        else:
            gain = min(0.9, gain)
        if abs(gain) < 0.35:
            skipped += 1
            continue
        item["gain_db"] = round(gain, 2)
        item["q"] = round(float(item["q"]), 2)
        by_channel.setdefault(channel, []).append(item)

    for channel, items in by_channel.items():
        # Keep this implementable on a real console: at most three anti-mask
        # moves per channel, with vocal-presence bands taking precedence.
        items.sort(key=lambda item: (
            0 if item["band"] in {"mid", "high_mid", "high"} else 1,
            -abs(float(item["gain_db"])),
        ))
        for item in items[:3]:
            plan = plans[channel]
            plan.eq_bands.append((
                float(item["frequency_hz"]),
                float(item["gain_db"]),
                float(item["q"]),
            ))
            plan.cross_adaptive_eq.append(dict(item))
            applied.append(dict(item))

    applied.sort(key=lambda item: (item["priority"], item["channel"], item["frequency_hz"]))
    return {
        "enabled": True,
        "raw_adjustments": len(raw_adjustments),
        "applied_adjustments": len(applied),
        "skipped_adjustments": skipped,
        "analysis_preview_sec": round(float(analysis_preview_sec), 2),
        "channel_priorities": {
            str(channel): priority
            for channel, priority in sorted(channel_priorities.items())
        },
        "notes": [
            "Cross-adaptive EQ is priority-driven: lower numeric priority means the channel is protected first.",
            "Lead vocals keep highest EQ priority, while competing accompaniment receives most anti-mask cuts.",
        ],
        "applied": applied,
    }


def classify_track(path: Path) -> tuple[str, float, float, float, list[tuple[float, float, float]], tuple[float, float, float, float], bool]:
    name = path.stem.lower()
    if "leadvoxdt" in name or "lead vox dt" in name:
        dt_idx = 1
        match = re.search(r"leadvoxdt(\d+)", name) or re.search(r"lead vox dt(\d+)", name)
        if match:
            dt_idx = int(match.group(1))
        pan = -0.14 if dt_idx % 2 == 1 else 0.14
        return "backing_vocal", pan, 110.0, -22.8, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.0, 6, 135), False
    if "backingvox" in name or "backing vox" in name:
        match = re.search(r"backingvox(\d+)", name) or re.search(r"backing vox(\d+)", name)
        bgv_idx = int(match.group(1)) if match else 1
        pan_map = {
            1: -0.52,
            2: 0.52,
            3: -0.28,
            4: 0.28,
            5: -0.66,
            6: 0.66,
        }
        pan = pan_map.get(bgv_idx, 0.0)
        if "dt" in name:
            pan = float(np.clip(-pan * 0.72 if abs(pan) > 0.01 else 0.22, -0.72, 0.72))
        return "backing_vocal", pan, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "kick" in name:
        return "kick", 0.0, 35.0, -20.0, [(60, 3.0, 0.9), (320, -3.0, 1.3), (4200, 2.0, 1.2)], (-18, 4.0, 8, 90), False
    if "snare bottom" in name or "snare b" in name:
        return "snare", 0.02, 120.0, -27.0, [(220, 1.0, 1.0), (5200, 2.0, 1.2), (850, -2.0, 2.0)], (-22, 3.5, 8, 110), True
    if "snare top" in name or name == "snare t" or name.startswith("snare t ") or name.endswith(" snare t") or name == "snaret":
        return "snare", -0.02, 90.0, -22.0, [(200, 2.0, 1.0), (850, -2.5, 2.0), (5200, 3.0, 1.0)], (-20, 4.0, 6, 120), False
    if "snare" in name or "snate" in name:
        return "snare", 0.0, 110.0, -23.0, [(220, -2.0, 1.1), (750, -2.5, 1.7), (4800, 2.0, 1.0)], (-19, 3.0, 8, 130), False
    if "floor" in name or name.startswith("f tom") or name.startswith("ftom"):
        return "floor_tom", 0.25, 55.0, -24.5, [(95, 2.0, 1.0), (360, -2.5, 1.5), (4200, 1.5, 1.2)], (-22, 3.0, 12, 150), False
    if "tom" in name:
        return "rack_tom", -0.20, 65.0, -25.0, [(120, 2.0, 1.0), (380, -2.0, 1.5), (4300, 1.5, 1.2)], (-22, 3.0, 12, 150), False
    if "hi hat" in name or "hi-hat" in name or "hihat" in name or name in {"hh", "hat"}:
        return "hi_hat", 0.0, 180.0, -28.0, [(450, -1.5, 1.2), (6500, 1.5, 1.0), (9500, 1.0, 0.8)], (-18, 1.8, 18, 180), False
    if "ride" in name:
        return "ride", 0.0, 170.0, -28.5, [(420, -1.2, 1.2), (4800, 1.2, 1.0), (9000, 1.0, 0.8)], (-18, 1.8, 20, 200), False
    if "overheads" in name or name == "overheads":
        return "overhead", 0.0, 150.0, -27.0, [(350, -1.5, 1.2), (3500, -1.0, 1.5), (10500, 1.5, 0.8)], (-18, 1.6, 25, 300), False
    if "overhead l" in name or name in {"oh l", "ohl"} or name.endswith(" oh l"):
        return "overhead", -0.72, 150.0, -27.0, [(350, -1.5, 1.2), (3500, -1.0, 1.5), (10500, 1.5, 0.8)], (-18, 1.6, 25, 300), False
    if "overhead r" in name or name in {"oh r", "ohr"} or name.endswith(" oh r"):
        return "overhead", 0.72, 150.0, -27.0, [(350, -1.5, 1.2), (3500, -1.0, 1.5), (10500, 1.5, 0.8)], (-18, 1.6, 25, 300), False
    if name.startswith("room dr l") or name.startswith("room l") or name in {"roomdr l", "room l"}:
        return "room", -0.78, 80.0, -31.0, [(250, -1.5, 1.2), (2500, -1.0, 1.4), (8500, 0.8, 1.0)], (-20, 2.0, 20, 220), False
    if name.startswith("room dr r") or name.startswith("room r") or name in {"roomdr r", "room r"}:
        return "room", 0.78, 80.0, -31.0, [(250, -1.5, 1.2), (2500, -1.0, 1.4), (8500, 0.8, 1.0)], (-20, 2.0, 20, 220), False
    if "room" in name:
        return "room", 0.0, 80.0, -31.0, [(250, -1.5, 1.2), (2500, -1.0, 1.4), (8500, 0.8, 1.0)], (-20, 2.0, 20, 220), False
    if "bass" in name:
        return "bass_guitar", 0.0, 35.0, -21.0, [(80, 2.0, 0.9), (250, -2.5, 1.2), (750, 1.2, 1.0)], (-24, 4.0, 18, 180), False
    if "elecgtr" in name or "eelcgtr" in name or "elec gtr" in name:
        match = re.search(r"(?:elecgtr|eelcgtr|elec gtr)(\d+)", name)
        gtr_idx = int(match.group(1)) if match else 1
        pan_map = {
            1: -0.68,
            2: 0.68,
            3: -0.38,
            4: 0.38,
            5: -0.18,
        }
        pan = pan_map.get(gtr_idx, -0.35 if gtr_idx % 2 else 0.35)
        if "dt" in name:
            pan = float(np.clip(-pan * 0.82 if abs(pan) > 0.01 else 0.22, -0.82, 0.82))
        return "electric_guitar", pan, 90.0, -25.5, [(250, -2.0, 1.2), (2500, 1.5, 1.0), (6200, -1.0, 1.0)], (-20, 2.0, 12, 140), False
    if "guitar l" in name:
        return "electric_guitar", -0.65, 90.0, -25.5, [(250, -2.0, 1.2), (2500, 1.5, 1.0), (6200, -1.0, 1.0)], (-20, 2.0, 12, 140), False
    if "guitar r" in name:
        return "electric_guitar", 0.65, 90.0, -25.5, [(250, -2.0, 1.2), (2500, 1.5, 1.0), (6200, -1.0, 1.0)], (-20, 2.0, 12, 140), False
    if "accordion" in name:
        return "accordion", -0.10, 100.0, -23.5, [(350, -1.5, 1.3), (2300, 1.3, 1.0), (7000, 0.8, 1.0)], (-21, 2.2, 15, 160), False
    if "playback l" in name:
        return "playback", -0.78, 30.0, -23.0, [(180, -0.8, 1.0), (3500, 0.8, 1.0)], (-16, 1.5, 25, 250), False
    if "playbacks r" in name or "playback r" in name:
        return "playback", 0.78, 30.0, -23.0, [(180, -0.8, 1.0), (3500, 0.8, 1.0)], (-16, 1.5, 25, 250), False
    if "back vox l" in name or "back vocal l" in name or "bvox l" in name or name in {"backs l", "back l", "bgv l"}:
        return "backing_vocal", -0.32, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "back vox r" in name or "back vocal r" in name or "bvox r" in name or name in {"backs r", "back r", "bgv r"}:
        return "backing_vocal", 0.32, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "back vox" in name or "back vocal" in name or "bvox" in name or "backs" in name or "bgv" in name:
        return "backing_vocal", 0.0, 110.0, -24.5, [(240, -2.0, 1.3), (2800, 1.5, 1.0), (9000, 1.0, 0.8)], (-24, 3.1, 6, 135), False
    if "vox" in name or "vocal" in name:
        return "lead_vocal", 0.0, 90.0, -18.8, [(250, -2.5, 1.4), (3100, 2.5, 1.0), (10500, 1.5, 0.8)], (-25, 4.4, 4, 115), False
    return "custom", 0.0, 80.0, -24.0, [(300, -1.0, 1.2), (3000, 0.8, 1.0)], (-20, 2.0, 10, 150), False


def highpass(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    if freq <= 0:
        return x
    b, a = butter(2, freq / (sr * 0.5), btype="highpass")
    return lfilter(b, a, x).astype(np.float32)


def lowpass(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    if freq <= 0 or freq >= sr * 0.49:
        return x
    b, a = butter(2, freq / (sr * 0.5), btype="lowpass")
    return lfilter(b, a, x).astype(np.float32)


def peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    if abs(gain_db) < 1e-4:
        return x
    a = db_to_amp(gain_db)
    w0 = 2.0 * math.pi * float(freq) / sr
    alpha = math.sin(w0) / (2.0 * max(float(q), 0.05))
    cos_w0 = math.cos(w0)
    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    aa = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, aa, x).astype(np.float32)


def _smooth_gain_reduction_samplewise(
    gain_reduction_db: np.ndarray,
    sr: int,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    attack = math.exp(-1.0 / max(1.0, attack_ms * 0.001 * sr))
    release = math.exp(-1.0 / max(1.0, release_ms * 0.001 * sr))
    smoothed = np.zeros_like(gain_reduction_db, dtype=np.float32)
    last = 0.0
    for i, gr in enumerate(gain_reduction_db):
        coeff = attack if gr > last else release
        last = coeff * last + (1.0 - coeff) * float(gr)
        smoothed[i] = last
    return smoothed


def _smooth_gain_reduction_fast(
    gain_reduction_db: np.ndarray,
    sr: int,
    attack_ms: float,
    release_ms: float,
    frame_ms: float = FAST_COMPRESSOR_FRAME_MS,
) -> np.ndarray:
    """Frame-rate compressor envelope.

    The old path smoothed gain reduction sample-by-sample in Python, which is
    expensive for 200-second multitracks. This keeps the same attack/release
    law, but evaluates it at a small control-rate frame and interpolates the
    gain envelope back to sample rate.
    """

    total_len = len(gain_reduction_db)
    if total_len == 0:
        return np.zeros(0, dtype=np.float32)

    frame = max(16, int(frame_ms * 0.001 * sr))
    frame = min(frame, total_len)
    padded_len = int(math.ceil(total_len / frame) * frame)
    padded = np.pad(
        np.asarray(gain_reduction_db, dtype=np.float32),
        (0, padded_len - total_len),
        mode="edge",
    )
    frame_view = padded.reshape(-1, frame)
    frame_mean = frame_view.mean(axis=1)
    frame_peak = frame_view.max(axis=1)
    frame_reduction = (frame_mean + (frame_peak - frame_mean) * 0.35).astype(np.float32)

    attack = math.exp(-frame / max(1.0, attack_ms * 0.001 * sr))
    release = math.exp(-frame / max(1.0, release_ms * 0.001 * sr))
    frame_smoothed = np.zeros_like(frame_reduction, dtype=np.float32)
    last = 0.0
    for i, gr in enumerate(frame_reduction):
        coeff = attack if gr > last else release
        last = coeff * last + (1.0 - coeff) * float(gr)
        frame_smoothed[i] = last

    sample_points = np.minimum(
        np.arange(len(frame_smoothed), dtype=np.float32) * frame + frame * 0.5,
        float(total_len - 1),
    )
    points = np.concatenate(([0.0], sample_points, [float(total_len - 1)]))
    values = np.concatenate(([frame_smoothed[0]], frame_smoothed, [frame_smoothed[-1]]))
    return np.interp(
        np.arange(total_len, dtype=np.float32),
        points,
        values,
    ).astype(np.float32)


def _zero_phase_filter(x: np.ndarray, sr: int, cutoff: float | tuple[float, float], btype: str) -> np.ndarray:
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    nyquist = sr * 0.5
    if btype == "lowpass":
        freq = float(cutoff)
        if freq <= 0.0 or freq >= nyquist * 0.98:
            return np.asarray(x, dtype=np.float32)
        wn = freq / nyquist
    elif btype == "highpass":
        freq = float(cutoff)
        if freq <= 0.0:
            return np.asarray(x, dtype=np.float32)
        wn = min(freq / nyquist, 0.98)
    else:
        low_hz, high_hz = cutoff
        low = max(1.0, float(low_hz))
        high = min(float(high_hz), nyquist * 0.98)
        if high <= low:
            return np.zeros_like(x, dtype=np.float32)
        wn = [low / nyquist, high / nyquist]
    b, a = butter(2, wn, btype=btype)
    try:
        return filtfilt(b, a, x).astype(np.float32)
    except Exception:
        return lfilter(b, a, x).astype(np.float32)


def _lowpass_zero_phase(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    return _zero_phase_filter(x, sr, freq, "lowpass")


def _highpass_zero_phase(x: np.ndarray, sr: int, freq: float) -> np.ndarray:
    return _zero_phase_filter(x, sr, freq, "highpass")


def _bandpass_zero_phase(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    return _zero_phase_filter(x, sr, (low_hz, high_hz), "bandpass")


def compressor(
    x: np.ndarray,
    sr: int,
    threshold_db: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    makeup_db: float = 0.0,
    auto_makeup: bool = True,
) -> np.ndarray:
    abs_x = np.maximum(np.abs(x), 1e-8)
    level_db = 20.0 * np.log10(abs_x)
    over = level_db - threshold_db
    gain_reduction_db = np.where(over > 0, over * (1.0 - 1.0 / max(ratio, 1.0)), 0.0)
    if len(gain_reduction_db) >= FAST_COMPRESSOR_MIN_SAMPLES and os.environ.get("AUTO_MIXER_SLOW_COMPRESSOR") != "1":
        smoothed = _smooth_gain_reduction_fast(gain_reduction_db, sr, attack_ms, release_ms)
    else:
        smoothed = _smooth_gain_reduction_samplewise(gain_reduction_db, sr, attack_ms, release_ms)
    gain = 10.0 ** (-smoothed / 20.0)
    compressed = (x * gain).astype(np.float32)

    total_makeup_db = float(makeup_db)
    if auto_makeup and len(compressed):
        input_rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
        output_rms = float(np.sqrt(np.mean(np.square(compressed))) + 1e-12)
        suppressed_db = max(0.0, amp_to_db(input_rms) - amp_to_db(output_rms))
        total_makeup_db += suppressed_db

    if abs(total_makeup_db) > 1e-4:
        compressed = (compressed * db_to_amp(total_makeup_db)).astype(np.float32)
    return compressed


def pan_mono(x: np.ndarray, pan: float) -> np.ndarray:
    left, right = _equal_power_gains(pan)
    return np.column_stack((x * left, x * right)).astype(np.float32)


def metrics_for(x: np.ndarray, sr: int, instrument: str | None = None) -> dict[str, Any]:
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    block, analysis_meta = _analysis_signal_for_metrics(x, sr, instrument)
    rms = float(np.sqrt(np.mean(np.square(block))) + 1e-12) if len(block) else 0.0
    freqs = np.fft.rfftfreq(min(len(block), sr * 8), 1.0 / sr)
    block = block[: len(freqs) * 2 - 2] if len(freqs) > 1 else block[:0]
    if len(block) > 0:
        spec = np.abs(np.fft.rfft(block * np.hanning(len(block)))) + 1e-12
        total = float(np.sum(spec))
        def band(lo: float, hi: float) -> float:
            idx = (freqs >= lo) & (freqs < hi)
            if not np.any(idx):
                return -100.0
            return amp_to_db(float(np.sum(spec[idx]) / total))
        band_energy = {
            "sub": band(20, 60),
            "bass": band(60, 250),
            "low_mid": band(250, 500),
            "mid": band(500, 2000),
            "presence": band(4000, 8000),
        }
    else:
        band_energy = {}
    return {
        "peak_db": amp_to_db(peak),
        "true_peak_db": amp_to_db(peak),
        "rms_db": amp_to_db(rms),
        "lufs_momentary": amp_to_db(rms),
        "dynamic_range_db": max(0.0, amp_to_db(peak) - amp_to_db(rms)),
        "band_energy": band_energy,
        "channel_armed": rms > db_to_amp(-55.0) or peak > db_to_amp(-45.0),
        "needs_attention": peak > db_to_amp(-6.0) or peak < db_to_amp(-30.0),
        "is_muted": False,
        **analysis_meta,
    }


def _repo_resolved_config_path(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def _make_offline_source_knowledge_layer(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[SourceKnowledgeLayer | None, dict[str, Any]]:
    """Create the optional source-grounded layer for offline candidate logging."""

    section = dict((config or {}).get("source_knowledge") or {})
    if getattr(args, "source_knowledge_enable", False):
        section["enabled"] = True
    if getattr(args, "source_knowledge_log", ""):
        section["log_path"] = str(args.source_knowledge_log)
    section.setdefault("mode", "shadow")
    if section.get("enabled"):
        section["queue_maxsize"] = max(1024, int(section.get("queue_maxsize", 256) or 256))
    for key in ("sources_path", "rules_path", "log_path"):
        if section.get(key):
            section[key] = _repo_resolved_config_path(section[key])

    try:
        layer = SourceKnowledgeLayer(SourceGroundedConfig.from_mapping(section))
        validation_errors = layer.store.validate()
        report = {
            "enabled": bool(layer.enabled),
            "mode": layer.config.mode,
            "log_path": str(layer.config.log_path),
            "sources_path": str(layer.store.sources_path),
            "rules_path": str(layer.store.rules_path),
            "validation_errors": validation_errors,
            "error": "",
        }
        if layer.enabled:
            layer.start()
        return layer, report
    except Exception as exc:
        return None, {
            "enabled": False,
            "mode": str(section.get("mode", "shadow")),
            "log_path": str(section.get("log_path", "")),
            "validation_errors": [],
            "error": str(exc),
        }


def _make_offline_perceptual_evaluator(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[PerceptualEvaluator | None, dict[str, Any]]:
    """Create the optional offline MERT/perceptual evaluator."""

    section = dict((config or {}).get("perceptual") or {})
    if getattr(args, "mert_enable", False):
        section.update({
            "enabled": True,
            "mode": "shadow",
            "backend": "mert",
            "async_evaluation": False,
            "log_scores": True,
            "evaluate_mix_bus": True,
            "evaluate_channels": False,
        })
    if getattr(args, "mert_model_name", ""):
        section["model_name"] = str(args.mert_model_name)
    if getattr(args, "mert_local_files_only", False):
        section["local_files_only"] = True
    if getattr(args, "perceptual_log", ""):
        section["log_path"] = str(args.perceptual_log)
    if getattr(args, "perceptual_window_sec", None) is not None:
        section["window_seconds"] = float(args.perceptual_window_sec)
    if section.get("log_path"):
        section["log_path"] = _repo_resolved_config_path(section["log_path"])

    try:
        evaluator = PerceptualEvaluator(PerceptualConfig.from_mapping(section))
        return evaluator, {
            "enabled": bool(evaluator.enabled),
            "mode": evaluator.mode,
            "requested_backend": str(section.get("backend", "lightweight")),
            "actual_backend": evaluator.backend.name,
            "model_name": str(section.get("model_name", "")),
            "sample_rate": int(evaluator.config.sample_rate),
            "window_seconds": float(evaluator.config.window_seconds),
            "log_path": str(evaluator.log_path),
            "error": "",
        }
    except Exception as exc:
        return None, {
            "enabled": False,
            "mode": str(section.get("mode", "shadow")),
            "requested_backend": str(section.get("backend", "lightweight")),
            "actual_backend": "",
            "model_name": str(section.get("model_name", "")),
            "log_path": str(section.get("log_path", "")),
            "error": str(exc),
        }


def _record_offline_perceptual_mix_bus(
    evaluator: PerceptualEvaluator | None,
    *,
    before_audio: np.ndarray,
    after_audio: np.ndarray,
    sr: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    if evaluator is None or not evaluator.enabled:
        return {"enabled": False}
    try:
        result = evaluator.record_shadow_decision(
            before_audio,
            after_audio,
            sr,
            context=context,
            osc_sent=False,
        )
        snapshot = evaluator.evaluate_mix_snapshot(
            after_audio,
            stems=None,
            context={"sample_rate": sr, **context},
        )
        payload = {
            "enabled": True,
            "backend": evaluator.backend.name,
            "decision": result.to_dict() if result is not None else None,
            "snapshot": snapshot,
            "error": "",
        }
        if result is not None:
            payload.update({
                "verdict": result.verdict,
                "delta_score": round(float(result.delta_score), 6),
                "mse": round(float(result.mse), 6),
                "cosine_distance": round(float(result.cosine_distance), 6),
                "confidence": round(float(result.confidence), 6),
                "reference_used": bool(result.reference_used),
            })
        return payload
    except Exception as exc:
        return {
            "enabled": True,
            "backend": getattr(getattr(evaluator, "backend", None), "name", ""),
            "decision": None,
            "snapshot": None,
            "error": str(exc),
        }


def _source_metrics(audio: np.ndarray, sr: int, instrument: str | None = None) -> dict[str, Any]:
    mono = mono_sum(np.asarray(audio, dtype=np.float32))
    raw = metrics_for(mono, sr, instrument=instrument)
    result: dict[str, Any] = {}
    for key in (
        "peak_db",
        "true_peak_db",
        "rms_db",
        "lufs_momentary",
        "dynamic_range_db",
        "analysis_active_sec",
        "analysis_active_ratio",
    ):
        if key in raw and isinstance(raw[key], (int, float, np.floating)):
            result[key] = round(float(raw[key]), 3)
    band_energy = raw.get("band_energy") or {}
    result["band_energy"] = {
        str(key): round(float(value), 3)
        for key, value in band_energy.items()
        if isinstance(value, (int, float, np.floating))
    }
    return result


def _source_metric_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("peak_db", "true_peak_db", "rms_db", "lufs_momentary", "dynamic_range_db"):
        if key in before and key in after:
            delta[key] = round(float(after[key]) - float(before[key]), 3)
    before_bands = before.get("band_energy") or {}
    after_bands = after.get("band_energy") or {}
    band_delta = {
        key: round(float(after_bands[key]) - float(before_bands[key]), 3)
        for key in before_bands.keys() & after_bands.keys()
    }
    if band_delta:
        delta["band_energy"] = band_delta
    return delta


def _source_rules_from_matches(
    matches: list[RuleMatch],
    selected_limit: int = 3,
) -> tuple[list[str], list[str], list[str]]:
    candidate_rule_ids = [match.rule.rule_id for match in matches]
    selected_rule_ids = candidate_rule_ids[:selected_limit]
    source_ids = sorted({
        source_id
        for match in matches[:selected_limit]
        for source_id in match.rule.source_ids
    })
    return candidate_rule_ids, selected_rule_ids, source_ids


def _safe_source_retrieve(
    layer: SourceKnowledgeLayer | None,
    query: str,
    *,
    domains: list[str],
    instrument: str,
    problems: list[str],
    action_types: list[str],
    limit: int = 6,
) -> list[RuleMatch]:
    if layer is None or not layer.enabled:
        return []
    try:
        matches = layer.retrieve(
            query,
            domains=domains,
            instruments=[instrument],
            problems=problems,
            action_types=action_types,
            limit=limit,
        )
        if matches:
            return matches
        return layer.retrieve(
            "bounded logged decisions source rule before after metrics feedback",
            domains=["automation", "logging"],
            instruments=["all"],
            problems=["training_data_gap"],
            action_types=["decision_trace"],
            limit=max(1, min(3, limit)),
        )
    except Exception:
        return []


def _infer_eq_problem(instrument: str, freq_hz: float, gain_db: float) -> tuple[list[str], str]:
    freq = float(freq_hz)
    gain = float(gain_db)
    if gain < 0.0 and 160.0 <= freq <= 520.0:
        return ["mud", "low_mid_buildup", "masking"], "low-mid cleanup"
    if gain < 0.0 and 2400.0 <= freq <= 8500.0:
        if "vocal" in instrument:
            return ["harshness", "sharp_vocal", "listener_fatigue"], "vocal harshness control"
        if "guitar" in instrument:
            return ["harshness", "sharp_guitar", "listener_fatigue"], "guitar harshness control"
        return ["harshness", "listener_fatigue"], "harshness control"
    if gain > 0.0 and freq < 140.0:
        return ["weak_low_end", "thin_low_end"], "low-end support"
    if gain > 0.0 and 1800.0 <= freq <= 6500.0:
        return ["thinness", "translation"], "presence support"
    return ["masking", "translation"], "tonal correction"


def _source_feedback_for_candidate(
    category: str,
    action: dict[str, Any],
    delta: dict[str, Any],
) -> tuple[str, str]:
    """Generate a conservative Codex listening proxy for later operator review."""

    if category == "eq":
        freq = float(action.get("freq_hz", 0.0) or 0.0)
        gain = float(action.get("gain_db", 0.0) or 0.0)
        if gain < 0.0 and 160.0 <= freq <= 520.0:
            return (
                "codex_predicted_better",
                "Expected listening result: less low-mid cloud without changing the musical role.",
            )
        if gain < 0.0 and 2400.0 <= freq <= 8500.0:
            return (
                "codex_predicted_better",
                "Expected listening result: smoother edge; verify articulation did not fall back.",
            )
        if gain > 0.0:
            return (
                "codex_watch",
                "Expected listening result: more definition; watch for added harshness or headroom loss.",
            )
    if category == "compressor":
        dr_delta = float(delta.get("dynamic_range_db", 0.0) or 0.0)
        if dr_delta < -0.2:
            return (
                "codex_predicted_better",
                "Expected listening result: steadier envelope and denser placement in the mix.",
            )
        return (
            "codex_neutral",
            "Expected listening result: subtle compression; operator A/B should confirm benefit.",
        )
    if category == "pan":
        pan = float(action.get("pan", 0.0) or 0.0)
        if abs(pan) < 0.05:
            return (
                "codex_neutral",
                "Expected listening result: center anchor preserved for mono compatibility.",
            )
        return (
            "codex_predicted_better",
            "Expected listening result: clearer soundfield role with less center masking.",
        )
    if category == "fx":
        return (
            "codex_watch",
            "Expected listening result: more depth or width; verify the wet return does not mask the front.",
        )
    return (
        "codex_neutral",
        "Expected listening result: logged for operator review.",
    )


def _source_candidate_confidence(matches: list[RuleMatch], action: dict[str, Any]) -> float:
    if not matches:
        return 0.0
    rule_conf = max(float(match.rule.confidence) for match in matches[:3])
    relevance = min(0.18, 0.03 * float(matches[0].relevance_score))
    magnitude = 0.0
    if "gain_db" in action:
        magnitude = min(0.08, abs(float(action.get("gain_db", 0.0))) / 40.0)
    elif "ratio" in action:
        magnitude = min(0.08, max(0.0, float(action.get("ratio", 1.0)) - 1.0) / 60.0)
    elif "pan" in action:
        magnitude = min(0.08, abs(float(action.get("pan", 0.0))) / 12.0)
    return round(float(np.clip(rule_conf + relevance + magnitude, 0.0, 0.98)), 3)


def _record_source_candidate(
    layer: SourceKnowledgeLayer | None,
    *,
    session_id: str,
    decision_id: str,
    channel: str,
    instrument: str,
    category: str,
    problem: str,
    matches: list[RuleMatch],
    action: dict[str, Any],
    before_audio: np.ndarray,
    after_audio: np.ndarray,
    sr: int,
    context: dict[str, Any] | None = None,
) -> bool:
    if layer is None or not layer.enabled:
        return False
    try:
        before_metrics = _source_metrics(before_audio, sr, instrument)
        after_metrics = _source_metrics(after_audio, sr, instrument)
        delta = _source_metric_delta(before_metrics, after_metrics)
        candidate_rule_ids, selected_rule_ids, source_ids = _source_rules_from_matches(matches)
        rating, comment = _source_feedback_for_candidate(category, action, delta)
        selected_action = dict(action)
        selected_action.update({
            "selected_rule_ids": selected_rule_ids,
            "source_ids": source_ids,
            "listening_feedback": comment,
        })
        trace_context = dict(context or {})
        trace_context.update({
            "offline_mix": True,
            "category": category,
            "metrics_delta": delta,
            "listening_feedback": {
                "listener": "codex",
                "rating": rating,
                "comment": comment,
            },
        })
        confidence = _source_candidate_confidence(matches, action)
        trace = DecisionTrace(
            session_id=session_id,
            decision_id=decision_id,
            channel=channel,
            instrument=instrument,
            problem=problem,
            context=trace_context,
            candidate_rule_ids=candidate_rule_ids,
            selected_rule_ids=selected_rule_ids,
            source_ids=source_ids,
            candidate_actions=[dict(action)],
            selected_action=selected_action,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            outcome=rating,
            confidence=confidence,
            osc_sent=False,
            safety_state={
                "mode": "shadow",
                "offline_only": True,
                "osc_behavior_changed": False,
            },
            notes=comment,
        )
        layer.record_decision(trace)
        layer.record_feedback(FeedbackRecord(
            session_id=session_id,
            decision_id=decision_id,
            listener="codex",
            rating=rating,
            comment=comment,
            preferred_action=selected_action,
            tags=["offline_mix", category, "listening_proxy"],
        ))
        return True
    except Exception:
        return False


SOURCE_RULES_ONLY_ANCHORS = {
    "kick",
    "snare",
    "bass",
    "bass_guitar",
    "bass_di",
    "bass_mic",
    "synth_bass",
    "lead_vocal",
}

SOURCE_RULES_ONLY_LEVEL_OFFSETS = {
    "lead_vocal": 1.6,
    "kick": 0.8,
    "bass": -0.1,
    "bass_guitar": -0.2,
    "bass_di": -0.2,
    "bass_mic": -0.2,
    "synth_bass": -0.4,
    "snare": -1.0,
    "rack_tom": -4.8,
    "floor_tom": -4.6,
    "hi_hat": -8.0,
    "ride": -8.0,
    "overhead": -8.4,
    "room": -9.2,
    "electric_guitar": -4.1,
    "lead_guitar": -3.7,
    "rhythm_guitar": -4.3,
    "acoustic_guitar": -4.7,
    "accordion": -5.0,
    "keys": -5.0,
    "piano": -5.0,
    "organ": -5.0,
    "synth": -5.2,
    "pad": -6.2,
    "playback": -5.5,
    "backing_vocal": -5.2,
}

SOURCE_RULES_ONLY_HPF = {
    "kick": 32.0,
    "bass": 35.0,
    "bass_guitar": 35.0,
    "bass_di": 35.0,
    "bass_mic": 35.0,
    "synth_bass": 28.0,
    "snare": 90.0,
    "rack_tom": 60.0,
    "floor_tom": 52.0,
    "hi_hat": 180.0,
    "ride": 170.0,
    "overhead": 170.0,
    "room": 85.0,
    "lead_vocal": 95.0,
    "backing_vocal": 110.0,
    "electric_guitar": 110.0,
    "lead_guitar": 100.0,
    "rhythm_guitar": 115.0,
    "acoustic_guitar": 100.0,
    "accordion": 125.0,
    "keys": 85.0,
    "piano": 80.0,
    "organ": 70.0,
    "synth": 55.0,
    "pad": 90.0,
    "playback": 35.0,
}


def _source_rule_ids(matches: list[RuleMatch], limit: int = 3) -> list[str]:
    return [match.rule.rule_id for match in matches[:limit]]


def _source_rules_only_retrieve(
    source_layer: SourceKnowledgeLayer | None,
    query: str,
    *,
    domains: list[str],
    instrument: str,
    problems: list[str],
    action_types: list[str],
    limit: int = 5,
) -> list[RuleMatch]:
    return _safe_source_retrieve(
        source_layer,
        query,
        domains=domains,
        instrument=instrument,
        problems=problems,
        action_types=action_types,
        limit=limit,
    )


def _source_rules_only_pan(plan: ChannelPlan) -> tuple[float, list[str]]:
    """Panning policy for the experimental source-rules-only offline render."""

    instrument = str(plan.instrument)
    name = plan.path.stem.lower()
    notes: list[str] = []
    if instrument in SOURCE_RULES_ONLY_ANCHORS:
        notes.append("anchor_kept_center")
        return 0.0, notes

    left_hint = bool(re.search(r"(^|[\s_.-])(l|left)([\s_.-]|$)", name))
    right_hint = bool(re.search(r"(^|[\s_.-])(r|right)([\s_.-]|$)", name))
    if left_hint and not right_hint:
        notes.append("filename_left_hint")
        direction = -1.0
    elif right_hint and not left_hint:
        notes.append("filename_right_hint")
        direction = 1.0
    else:
        stable_hash = sum(ord(char) for char in plan.path.stem.lower())
        direction = -1.0 if stable_hash % 2 else 1.0
        notes.append("role_based_alternating_support_pan")

    width_by_instrument = {
        "overhead": 0.78,
        "room": 0.72,
        "playback": 0.70,
        "electric_guitar": 0.62,
        "lead_guitar": 0.58,
        "rhythm_guitar": 0.62,
        "acoustic_guitar": 0.42,
        "backing_vocal": 0.38,
        "accordion": 0.34,
        "keys": 0.40,
        "piano": 0.34,
        "organ": 0.34,
        "synth": 0.44,
        "pad": 0.58,
        "hi_hat": 0.28,
        "ride": 0.28,
        "rack_tom": 0.22,
        "floor_tom": 0.28,
    }
    pan = float(direction * width_by_instrument.get(instrument, 0.25))
    return float(np.clip(pan, -0.85, 0.85)), notes


def _source_rules_only_band_medians(plans: dict[int, ChannelPlan]) -> dict[str, float]:
    medians: dict[str, float] = {}
    for band in ("sub", "bass", "low_mid", "mid", "presence"):
        values = []
        for plan in plans.values():
            band_energy = plan.metrics.get("band_energy") or {}
            value = band_energy.get(band)
            if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
                values.append(float(value))
        medians[band] = _median_value(values, default=-24.0)
    return medians


def _source_rules_only_eq_bands(
    plan: ChannelPlan,
    band_medians: dict[str, float],
    source_layer: SourceKnowledgeLayer | None,
) -> tuple[list[tuple[float, float, float]], list[dict[str, Any]]]:
    instrument = str(plan.instrument)
    bands = plan.metrics.get("band_energy") or {}
    decisions: list[dict[str, Any]] = []
    eq: list[tuple[float, float, float]] = []

    def add_band(
        freq: float,
        gain: float,
        q: float,
        problem: str,
        query: str,
        domains: list[str] | None = None,
    ) -> None:
        matches = _source_rules_only_retrieve(
            source_layer,
            query,
            domains=domains or ["eq", "masking", "tone"],
            instrument=instrument,
            problems=[problem, "masking", "translation"],
            action_types=["eq_candidate"],
            limit=4,
        )
        eq.append((float(freq), float(gain), float(q)))
        decisions.append({
            "problem": problem,
            "freq_hz": round(float(freq), 2),
            "gain_db": round(float(gain), 2),
            "q": round(float(q), 2),
            "selected_rule_ids": _source_rule_ids(matches),
        })

    low_mid = float(bands.get("low_mid", band_medians.get("low_mid", -24.0)))
    mid = float(bands.get("mid", band_medians.get("mid", -24.0)))
    presence = float(bands.get("presence", band_medians.get("presence", -30.0)))
    bass = float(bands.get("bass", band_medians.get("bass", -20.0)))
    low_mid_excess = low_mid - float(band_medians.get("low_mid", low_mid))
    presence_excess = presence - float(band_medians.get("presence", presence))

    if low_mid_excess > 1.3:
        cut_db = -float(np.clip(1.0 + low_mid_excess * 0.34, 1.1, 2.8))
        center = 320.0
        if instrument in {"bass", "bass_guitar", "bass_di", "bass_mic"}:
            center = 210.0
        elif instrument in {"snare", "rack_tom", "floor_tom"}:
            center = 360.0
        elif "guitar" in instrument:
            center = 280.0
        add_band(
            center,
            cut_db,
            1.2,
            "low_mid_buildup",
            f"{instrument} low mid mud masking measured excess",
            domains=["eq", "masking"],
        )

    if presence_excess > 2.2 and instrument in {
        "lead_vocal",
        "backing_vocal",
        "electric_guitar",
        "lead_guitar",
        "rhythm_guitar",
        "accordion",
        "hi_hat",
        "ride",
        "overhead",
    }:
        center = 3600.0 if "vocal" in instrument else 4200.0
        if instrument in {"hi_hat", "ride", "overhead"}:
            center = 6200.0
        cut_db = -float(np.clip(0.8 + presence_excess * 0.22, 0.9, 2.2))
        add_band(
            center,
            cut_db,
            1.5,
            "harshness",
            f"{instrument} measured presence harshness targeted cut",
            domains=["eq", "harshness"],
        )

    if instrument == "kick":
        click_shortfall = (bass - presence)
        if click_shortfall > 15.0:
            add_band(
                3600.0,
                float(np.clip((click_shortfall - 14.0) * 0.16, 0.8, 2.2)),
                1.25,
                "weak_attack",
                "kick weak click attack measured bass dominates presence",
                domains=["eq", "dynamics", "drums"],
            )
        if low_mid > bass - 1.0:
            add_band(
                320.0,
                -1.4,
                1.25,
                "boxiness",
                "kick boxy low mid measured cleanup",
                domains=["eq", "masking", "drums"],
            )
    elif instrument == "snare":
        if mid - presence > 1.0:
            add_band(
                4800.0,
                1.0,
                1.15,
                "weak_attack",
                "snare attack presence measured support",
                domains=["eq", "drums"],
            )
        if low_mid_excess > 0.8:
            add_band(
                520.0,
                -1.0,
                1.7,
                "boxiness",
                "snare boxy midrange measured cleanup",
                domains=["eq", "masking", "drums"],
            )
    elif instrument == "lead_vocal":
        if presence_excess < -1.8:
            add_band(
                2800.0,
                1.0,
                1.0,
                "vocal_intelligibility",
                "lead vocal measured presence support",
                domains=["eq", "vocal", "tone"],
            )
        if low_mid_excess > 0.5:
            add_band(
                240.0,
                -1.2,
                1.2,
                "low_mid_buildup",
                "lead vocal low mid cleanup measured proximity",
                domains=["eq", "masking", "vocal"],
            )

    return eq[:4], decisions


def _source_rules_only_compressor(
    plan: ChannelPlan,
    source_layer: SourceKnowledgeLayer | None,
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    instrument = str(plan.instrument)
    metrics = plan.metrics or {}
    rms_db = float(metrics.get("rms_db", -36.0))
    dynamic_range = float(metrics.get("dynamic_range_db", 0.0))
    active_ratio = float(metrics.get("analysis_active_ratio", 1.0) or 1.0)
    matches = _source_rules_only_retrieve(
        source_layer,
        f"{instrument} compression envelope dynamics measured {dynamic_range:.1f} dB",
        domains=["dynamics", "compression"],
        instrument=instrument,
        problems=["unstable_level", "excessive_peaks", "weak_attack", "flat_dynamics"],
        action_types=["compressor_candidate"],
        limit=5,
    )

    threshold = 0.0
    ratio = 1.0
    attack = 18.0
    release = 150.0
    reason = "left_uncompressed_by_metrics"
    if instrument == "kick" and dynamic_range > 7.5:
        threshold = float(np.clip(rms_db + 4.0, -28.0, -8.0))
        ratio = 3.0
        attack = 24.0
        release = 95.0
        reason = "kick_transient_preserving_compression"
    elif instrument == "snare" and dynamic_range > 8.0:
        threshold = float(np.clip(rms_db + 3.0, -30.0, -10.0))
        ratio = 3.3
        attack = 7.0
        release = 125.0
        reason = "snare_body_without_losing_attack"
    elif instrument in {"rack_tom", "floor_tom"} and dynamic_range > 9.0 and active_ratio > 0.02:
        threshold = float(np.clip(rms_db + 4.0, -30.0, -9.0))
        ratio = 2.5
        attack = 16.0
        release = 145.0
        reason = "tom_level_density"
    elif instrument in {"lead_vocal", "backing_vocal"} and dynamic_range > 8.0:
        threshold = float(np.clip(rms_db + 2.0, -32.0, -10.0))
        ratio = 2.7 if instrument == "lead_vocal" else 2.0
        attack = 9.0
        release = 135.0
        reason = "vocal_consistency_from_measured_dynamics"
    elif instrument in BASS_INSTRUMENTS and dynamic_range > 7.0:
        threshold = float(np.clip(rms_db + 2.5, -30.0, -10.0))
        ratio = 3.0
        attack = 18.0
        release = 165.0
        reason = "bass_note_stability"
    elif "guitar" in instrument and dynamic_range > 12.0:
        threshold = float(np.clip(rms_db + 4.0, -30.0, -10.0))
        ratio = 1.7
        attack = 14.0
        release = 145.0
        reason = "support_source_peak_control"
    elif instrument in {"accordion", "keys", "piano", "organ", "synth", "playback"} and dynamic_range > 13.0:
        threshold = float(np.clip(rms_db + 4.0, -30.0, -10.0))
        ratio = 1.6
        attack = 18.0
        release = 165.0
        reason = "support_layer_peak_control"

    return (
        (threshold, ratio, attack, release),
        {
            "reason": reason,
            "dynamic_range_db": round(dynamic_range, 2),
            "analysis_active_ratio": round(active_ratio, 4),
            "selected_rule_ids": _source_rule_ids(matches),
        },
    )


def apply_source_rules_mert_only_plan(
    plans: dict[int, ChannelPlan],
    sr: int,
    target_len: int,
    source_layer: SourceKnowledgeLayer | None,
    source_session_id: str,
) -> dict[str, Any]:
    """Build an offline mix plan from metrics, source rules, and MERT shadow only.

    This intentionally bypasses the project's normal rule engine, AutoFOH analyzers,
    reference voicing, and agent actions. The final limiter/headroom stage remains
    outside this function as a non-negotiable offline render safety guard.
    """

    active_rms = [
        float(plan.metrics.get("rms_db", -60.0))
        for plan in plans.values()
        if float(plan.metrics.get("rms_db", -120.0)) > -65.0
    ]
    base_target = float(np.clip(_median_value(active_rms, default=-25.0) - 1.0, -29.0, -21.0))
    band_medians = _source_rules_only_band_medians(plans)
    level_matches = _source_rules_only_retrieve(
        source_layer,
        "automatic balance loudness metrics role priority true peak",
        domains=["balance", "fader", "metering"],
        instrument="all",
        problems=["unstable_starting_point", "unknown_loudness", "unclear_mix"],
        action_types=["balance_pass"],
        limit=5,
    )
    pan_matches = _source_rules_only_retrieve(
        source_layer,
        "panning unmask supports keep anchors center",
        domains=["pan", "soundfield", "masking"],
        instrument="all",
        problems=["crowded_center", "narrow_mix", "masking"],
        action_types=["pan_candidate"],
        limit=5,
    )
    hpf_matches = _source_rules_only_retrieve(
        source_layer,
        "filtered low end cleanup source rule hpf masking mud",
        domains=["eq", "filter", "masking"],
        instrument="all",
        problems=["mud", "masking", "headroom_risk"],
        action_types=["eq_candidate"],
        limit=5,
    )

    channel_reports: list[dict[str, Any]] = []
    selected_rule_ids: set[str] = set()
    for match in [*level_matches, *pan_matches, *hpf_matches]:
        selected_rule_ids.add(match.rule.rule_id)

    for channel, plan in plans.items():
        instrument = str(plan.instrument)
        metrics = plan.metrics or {}
        rms_db = float(metrics.get("rms_db", -60.0))
        active_ratio = float(metrics.get("analysis_active_ratio", 1.0) or 1.0)
        is_bleed_like = bool(metrics.get("analysis_bleed_dominant", False)) or (
            instrument in {"overhead", "room", "hi_hat", "ride"}
        )

        target = base_target + float(SOURCE_RULES_ONLY_LEVEL_OFFSETS.get(instrument, -4.5))
        if "snare" in plan.path.stem.lower() and ("bottom" in plan.path.stem.lower() or re.search(r"\bsnare\s*b\b", plan.path.stem.lower())):
            target -= 7.0
        max_boost = 8.5
        if is_bleed_like:
            max_boost = 3.5
        elif active_ratio < 0.04 and instrument in {"rack_tom", "floor_tom", "snare"}:
            max_boost = 5.5
        trim_raw = target - rms_db
        trim = float(np.clip(trim_raw, -18.0, max_boost))
        pan, pan_notes = _source_rules_only_pan(plan)
        eq_bands, eq_decisions = _source_rules_only_eq_bands(plan, band_medians, source_layer)
        (threshold, ratio, attack, release), comp_decision = _source_rules_only_compressor(plan, source_layer)

        plan.target_rms_db = float(target)
        plan.trim_db = trim
        plan.fader_db = 0.0
        plan.pan = pan
        plan.hpf = float(SOURCE_RULES_ONLY_HPF.get(instrument, 80.0))
        plan.lpf = 0.0
        plan.eq_bands = eq_bands
        plan.comp_threshold_db = float(threshold)
        plan.comp_ratio = float(ratio)
        plan.comp_attack_ms = float(attack)
        plan.comp_release_ms = float(release)
        plan.phase_invert = False
        plan.delay_ms = 0.0
        plan.expander_enabled = False
        plan.expander_range_db = 0.0
        plan.expander_report = {
            "enabled": False,
            "reason": "source_rules_mert_only_disables_project_event_expander",
        }
        plan.phase_notes = [{
            "mode": "source_rules_mert_only",
            "decision": "no_phase_or_delay_change",
            "reason": "phase rule is shadow/advisory unless auditioned against full mix",
        }]
        plan.pan_notes = [{
            "mode": "source_rules_mert_only",
            "pan": round(float(pan), 3),
            "notes": pan_notes,
            "selected_rule_ids": _source_rule_ids(pan_matches),
        }]
        plan.cross_adaptive_eq = []
        plan.autofoh_actions = []
        plan.fx_send_db = None
        plan.fx_bus_send_db = {}
        plan.trim_analysis = {
            "mode": "source_rules_mert_only",
            "base_target_rms_db": round(base_target, 2),
            "target_rms_db": round(float(target), 2),
            "input_rms_db": round(rms_db, 2),
            "raw_trim_db": round(float(trim_raw), 2),
            "applied_trim_db": round(float(trim), 2),
            "max_boost_db": round(float(max_boost), 2),
            "analysis_active_ratio": round(active_ratio, 4),
            "bleed_like": bool(is_bleed_like),
            "selected_rule_ids": _source_rule_ids(level_matches),
        }

        for item in eq_decisions:
            selected_rule_ids.update(item.get("selected_rule_ids", []))
        selected_rule_ids.update(comp_decision.get("selected_rule_ids", []))

        channel_reports.append({
            "channel": int(channel),
            "file": plan.path.name,
            "instrument": instrument,
            "target_rms_db": round(float(target), 2),
            "trim_db": round(float(trim), 2),
            "fader_db": 0.0,
            "pan": round(float(pan), 3),
            "hpf_hz": round(float(plan.hpf), 1),
            "eq": eq_decisions,
            "compressor": {
                "threshold_db": round(float(threshold), 2),
                "ratio": round(float(ratio), 2),
                "attack_ms": round(float(attack), 2),
                "release_ms": round(float(release), 2),
                **comp_decision,
            },
            "phase": "shadow_only_no_change",
        })

    return {
        "enabled": True,
        "mode": "source_rules_mert_only",
        "sample_rate": int(sr),
        "duration_sec": round(float(target_len) / float(sr), 3) if sr else 0.0,
        "source_session_id": source_session_id,
        "base_target_rms_db": round(base_target, 2),
        "band_medians": {key: round(float(value), 3) for key, value in band_medians.items()},
        "selected_rule_ids": sorted(selected_rule_ids),
        "disabled_project_passes": [
            "mixing_agent_rule_engine",
            "codex_orchestrator",
            "drum_phase_alignment",
            "overhead_anchored_drum_panning",
            "event_based_dynamics",
            "autofoh_analyzer_pass",
            "bass_drum_push",
            "kick_presence_boost",
            "cymbal_focus_cleanup",
            "cross_adaptive_eq",
            "reference_mix_guidance",
            "genre_mix_profile",
            "kick_bass_hierarchy",
            "frequency_window_balance",
            "stem_mix_verification",
            "reference_vocal_fx_focus",
            "reference_dynamics_ride",
            "large_system_polish",
            "master_bus_glue_compressor",
        ],
        "safety_kept": [
            "offline_render_only_no_osc",
            "channel_faders_not_above_0_db",
            "final_limiter_and_master_ceiling",
            "mert_shadow_only_no_command_blocking",
        ],
        "channels": channel_reports,
    }


def build_source_rules_only_fx_plan(
    plans: dict[int, ChannelPlan],
    tempo_bpm: float = 120.0,
) -> FXPlan:
    """Create a source-rules-only FX plan without invoking AutoFXPlanner."""

    quarter_ms = 60000.0 / max(40.0, min(float(tempo_bpm), 240.0))
    dotted_eighth_ms = quarter_ms * 0.75
    quarter_delay_ms = quarter_ms
    buses = [
        FXBusDecision(
            bus_id=13,
            name="Source Plate",
            fx_type="reverb",
            fx_slot="aux_13",
            model="source_rules_filtered_plate",
            return_level_db=-7.5,
            params={
                "decay_s": 1.35,
                "predelay_ms": 48.0,
                "density": 0.62,
                "brightness": 0.42,
            },
            hpf_hz=220.0,
            lpf_hz=7200.0,
            duck_source="lead_vocal",
            duck_depth_db=2.0,
            reason="rules_jsonl filtered shared vocal depth with predelay",
        ),
        FXBusDecision(
            bus_id=14,
            name="Source Drum Room",
            fx_type="reverb",
            fx_slot="aux_14",
            model="source_rules_short_room",
            return_level_db=-9.5,
            params={
                "decay_s": 0.62,
                "predelay_ms": 14.0,
                "density": 0.78,
                "brightness": 0.36,
            },
            hpf_hz=180.0,
            lpf_hz=5600.0,
            duck_source=None,
            duck_depth_db=0.0,
            reason="rules_jsonl short filtered drum ambience preserving attack",
        ),
        FXBusDecision(
            bus_id=15,
            name="Source Tempo Delay",
            fx_type="delay",
            fx_slot="aux_15",
            model="source_rules_filtered_tempo_delay",
            return_level_db=-5.5,
            params={
                "left_delay_ms": float(np.clip(dotted_eighth_ms, 120.0, 500.0)),
                "right_delay_ms": float(np.clip(quarter_delay_ms, 150.0, 500.0)),
                "feedback": 0.18,
                "width": 0.72,
            },
            hpf_hz=260.0,
            lpf_hz=5200.0,
            duck_source="lead_vocal",
            duck_depth_db=4.0,
            reason="rules_jsonl filtered ducked delay for depth without reverb wash",
        ),
        FXBusDecision(
            bus_id=16,
            name="Source Mod Doubler",
            fx_type="chorus",
            fx_slot="aux_16",
            model="source_rules_filtered_chorus_width",
            return_level_db=-13.0,
            params={
                "left_delay_ms": 11.0,
                "right_delay_ms": 17.0,
                "depth": 0.13,
                "rate_hz": 0.45,
            },
            hpf_hz=220.0,
            lpf_hz=8500.0,
            duck_source=None,
            duck_depth_db=0.0,
            reason="rules_jsonl subtle support-source modulation width",
        ),
    ]
    sends: list[FXSendDecision] = []
    for channel, plan in plans.items():
        if plan.muted:
            continue
        instrument = str(plan.instrument)

        def add(bus_id: int, send_db: float, reason: str) -> None:
            sends.append(FXSendDecision(
                channel_id=int(channel),
                instrument=instrument,
                bus_id=int(bus_id),
                send_db=float(send_db),
                post_fader=True,
                reason=reason,
            ))

        if instrument == "lead_vocal":
            add(13, -18.5, "filtered plate keeps lead present but not dry")
            add(15, -25.0, "ducked tempo delay adds vocal depth")
        elif instrument == "backing_vocal":
            add(13, -16.0, "backing vocals placed deeper than lead")
            add(16, -24.5, "subtle modulation widens support vocal")
        elif instrument == "snare":
            add(14, -18.5, "short room restores snare space without long wash")
            add(13, -27.0, "small shared plate tail")
        elif instrument in {"rack_tom", "floor_tom"}:
            add(14, -20.5, "short room supports tom body")
        elif instrument in {"overhead", "room", "hi_hat", "ride"}:
            add(14, -34.0, "minimal room return because source already contains kit ambience")
        elif "guitar" in instrument:
            add(13, -25.5, "filtered plate gives guitar depth")
            add(16, -27.0, "subtle modulation keeps guitar off the center")
        elif instrument in {"accordion", "keys", "piano", "organ", "synth", "pad"}:
            add(13, -24.5, "filtered shared space for support instrument")
            add(15, -30.0, "quiet delay depth for support instrument")
            add(16, -28.0, "light modulation texture")
        elif instrument == "playback":
            add(13, -31.0, "very low shared space on prebuilt layer")
            add(16, -31.0, "very low support width on prebuilt layer")

    return FXPlan(
        buses=buses,
        sends=sends,
        notes=[
            "source_rules_mert_only",
            "Built without AutoFXPlanner; values come from source FX rules and measured role classification.",
        ],
    )


def _layer_role_from_name(plan: ChannelPlan) -> str:
    name = plan.path.stem.lower()
    if "bottom" in name or re.search(r"\bsnare\s*b\b", name) or name.endswith(" b"):
        return "bottom"
    if "top" in name or re.search(r"\bsnare\s*t\b", name) or name.endswith(" t"):
        return "top"
    if re.search(r"(^|[\s_.-])(l|left)([\s_.-]|$)", name):
        return "left"
    if re.search(r"(^|[\s_.-])(r|right)([\s_.-]|$)", name):
        return "right"
    if "dt" in name or "double" in name:
        return "double"
    return "main"


def _layer_group_key(plan: ChannelPlan) -> tuple[str, str, str] | None:
    instrument = str(plan.instrument)
    role = _layer_role_from_name(plan)
    if instrument == "snare":
        return ("snare", "snare", "top_bottom")
    if instrument in {"overhead", "room", "playback"} and role in {"left", "right"}:
        return (f"{instrument}_stereo", instrument, "stereo_pair")
    if instrument == "backing_vocal" and role in {"left", "right", "double"}:
        return ("backing_vocal_stack", instrument, "support_stack")
    if "guitar" in instrument and role in {"left", "right", "double"}:
        return ("electric_guitar_layers", instrument, "support_stack")
    return None


def build_layer_group_plans(plans: dict[int, ChannelPlan]) -> list[LayerGroupPlan]:
    groups: dict[str, LayerGroupPlan] = {}
    for channel, plan in sorted(plans.items()):
        if plan.muted:
            continue
        key = _layer_group_key(plan)
        if key is None:
            continue
        group_id, instrument, group_kind = key
        group = groups.setdefault(
            group_id,
            LayerGroupPlan(
                group_id=group_id,
                instrument=instrument,
                channels=[],
                roles={},
                group_kind=group_kind,
            ),
        )
        group.channels.append(int(channel))
        group.roles[int(channel)] = _layer_role_from_name(plan)
    return [group for group in groups.values() if len(group.channels) > 1]


def _audio_rms_db(audio: np.ndarray) -> float:
    mono = mono_sum(np.asarray(audio, dtype=np.float32))
    if len(mono) == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(mono))) + 1e-12)
    return amp_to_db(rms)


def _audio_peak_db(audio: np.ndarray) -> float:
    arr = np.asarray(audio, dtype=np.float32)
    if len(arr) == 0:
        return -120.0
    return amp_to_db(float(np.max(np.abs(arr))))


def _apply_stereo_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return (np.asarray(audio, dtype=np.float32) * db_to_amp(float(gain_db))).astype(np.float32)


def _apply_stereo_peaking_eq(audio: np.ndarray, sr: int, freq: float, gain_db: float, q: float) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 1:
        return peaking_eq(arr, sr, freq, gain_db, q)
    return np.column_stack([
        peaking_eq(arr[:, channel], sr, freq, gain_db, q)
        for channel in range(arr.shape[1])
    ]).astype(np.float32)


def _sum_layer_group_audio(
    rendered_channels: dict[int, np.ndarray],
    group: LayerGroupPlan,
) -> np.ndarray:
    first = rendered_channels[group.channels[0]]
    return sum((rendered_channels[channel] for channel in group.channels), np.zeros_like(first))


def _layer_group_phase_score(audio: np.ndarray, sr: int, instrument: str) -> float:
    mono = mono_sum(audio)
    if len(mono) == 0:
        return -120.0
    if instrument == "snare":
        body = _bandpass_zero_phase(mono, sr, 140.0, 520.0)
        crack = _bandpass_zero_phase(mono, sr, 1800.0, 6500.0)
        body_db = _audio_rms_db(body)
        crack_db = _audio_rms_db(crack)
        peak_db = _audio_peak_db(mono)
        return float(body_db * 0.68 + crack_db * 0.22 + peak_db * 0.10)
    if instrument in BASS_INSTRUMENTS or instrument == "kick":
        low = _bandpass_zero_phase(mono, sr, 45.0, 180.0)
        return _audio_rms_db(low)
    return _audio_rms_db(mono)


def _apply_layer_render_gain(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    channel: int,
    gain_db: float,
    reason: str,
) -> None:
    if abs(float(gain_db)) < 1e-4:
        return
    rendered_channels[channel] = _apply_stereo_gain(rendered_channels[channel], gain_db)
    plans[channel].trim_db = float(np.clip(plans[channel].trim_db + float(gain_db), -30.0, 12.0))
    plans[channel].trim_analysis.setdefault("layer_group_adjustments", []).append({
        "gain_db": round(float(gain_db), 3),
        "reason": reason,
    })


def _apply_layer_group_internal_balance(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    group: LayerGroupPlan,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    levels = {channel: _audio_rms_db(rendered_channels[channel]) for channel in group.channels}
    if group.group_kind == "top_bottom":
        top_channels = [ch for ch in group.channels if group.roles.get(ch) == "top"]
        if not top_channels:
            top_channels = [
                ch for ch in group.channels
                if group.roles.get(ch) not in {"bottom"}
            ]
        if not top_channels:
            return actions
        top_channel = max(top_channels, key=lambda ch: levels.get(ch, -120.0))
        top_level = levels.get(top_channel, -120.0)
        for channel in group.channels:
            if group.roles.get(channel) != "bottom":
                continue
            before = levels.get(channel, -120.0)
            target = top_level - 10.0
            gain = float(np.clip(target - before, -12.0, 3.0))
            _apply_layer_render_gain(
                rendered_channels,
                plans,
                channel,
                gain,
                "layer_group_internal_snare_bottom_to_top_minus_10_db",
            )
            after = _audio_rms_db(rendered_channels[channel])
            actions.append({
                "type": "internal_balance",
                "channel": int(channel),
                "file": plans[channel].path.name,
                "role": "bottom",
                "reference_channel": int(top_channel),
                "reference_file": plans[top_channel].path.name,
                "target_relative_db": -10.0,
                "before_rms_db": round(before, 2),
                "after_rms_db": round(after, 2),
                "gain_db": round(gain, 2),
            })
        return actions

    if group.group_kind in {"stereo_pair", "support_stack"}:
        usable = [
            levels[channel]
            for channel in group.channels
            if np.isfinite(levels.get(channel, -120.0)) and levels[channel] > -90.0
        ]
        if len(usable) < 2:
            return actions
        target = _median_value(usable, default=usable[0])
        for channel in group.channels:
            before = levels[channel]
            gain = float(np.clip(target - before, -3.0, 3.0))
            if abs(gain) < 0.25:
                continue
            _apply_layer_render_gain(
                rendered_channels,
                plans,
                channel,
                gain,
                "layer_group_internal_pair_balance",
            )
            actions.append({
                "type": "internal_balance",
                "channel": int(channel),
                "file": plans[channel].path.name,
                "role": group.roles.get(channel, "layer"),
                "target_rms_db": round(target, 2),
                "before_rms_db": round(before, 2),
                "after_rms_db": round(_audio_rms_db(rendered_channels[channel]), 2),
                "gain_db": round(gain, 2),
            })
    return actions


def _apply_layer_group_phase(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    group: LayerGroupPlan,
    sr: int,
) -> list[dict[str, Any]]:
    if group.group_kind != "top_bottom":
        return []
    actions: list[dict[str, Any]] = []
    for channel in group.channels:
        if group.roles.get(channel) != "bottom":
            continue
        before_audio = _sum_layer_group_audio(rendered_channels, group)
        before_score = _layer_group_phase_score(before_audio, sr, group.instrument)
        rendered_channels[channel] = (-rendered_channels[channel]).astype(np.float32)
        flipped_audio = _sum_layer_group_audio(rendered_channels, group)
        flipped_score = _layer_group_phase_score(flipped_audio, sr, group.instrument)
        improvement = float(flipped_score - before_score)
        if improvement >= 0.35:
            plans[channel].phase_invert = not bool(plans[channel].phase_invert)
            action = {
                "type": "phase",
                "channel": int(channel),
                "file": plans[channel].path.name,
                "role": group.roles.get(channel),
                "decision": "polarity_flipped",
                "score_before": round(before_score, 3),
                "score_after": round(flipped_score, 3),
                "improvement_db": round(improvement, 3),
            }
        else:
            rendered_channels[channel] = (-rendered_channels[channel]).astype(np.float32)
            action = {
                "type": "phase",
                "channel": int(channel),
                "file": plans[channel].path.name,
                "role": group.roles.get(channel),
                "decision": "kept_original_polarity",
                "score_before": round(before_score, 3),
                "score_after_if_flipped": round(flipped_score, 3),
                "improvement_db": round(improvement, 3),
            }
        plans[channel].phase_notes.append({
            "mode": "layer_group_sum",
            **action,
        })
        actions.append(action)
    return actions


def _layer_group_target_rms(base_target_rms_db: float, instrument: str) -> float:
    return float(base_target_rms_db + SOURCE_RULES_ONLY_LEVEL_OFFSETS.get(instrument, -4.5))


def _infer_layer_group_base_target(plans: dict[int, ChannelPlan]) -> float:
    """Infer a group-level balance base from the current mix plan."""

    candidates: list[float] = []
    for plan in plans.values():
        if plan.muted:
            continue
        try:
            target = float(plan.target_rms_db)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(target):
            continue
        offset = float(SOURCE_RULES_ONLY_LEVEL_OFFSETS.get(str(plan.instrument), -4.5))
        candidates.append(target - offset)
    return float(np.clip(_median_value(candidates, default=-24.5), -29.0, -18.0))


def _apply_layer_group_sum_corrections(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    group: LayerGroupPlan,
    sr: int,
    base_target_rms_db: float,
    band_medians: dict[str, float],
    source_layer: SourceKnowledgeLayer | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    before_audio = _sum_layer_group_audio(rendered_channels, group)
    before_metrics = _source_metrics(before_audio, sr, group.instrument)
    level_matches = _source_rules_only_retrieve(
        source_layer,
        f"{group.instrument} layered instrument summed loudness balance",
        domains=["balance", "fader", "metering"],
        instrument=group.instrument,
        problems=["unstable_starting_point", "unknown_loudness", "unclear_mix"],
        action_types=["balance_pass"],
        limit=5,
    )
    eq_matches = _source_rules_only_retrieve(
        source_layer,
        f"{group.instrument} layered sum low mid masking eq",
        domains=["eq", "masking"],
        instrument=group.instrument,
        problems=["low_mid_buildup", "mud", "masking", "boxiness"],
        action_types=["eq_candidate"],
        limit=5,
    )
    pan_matches = _source_rules_only_retrieve(
        source_layer,
        f"{group.instrument} layered sum position pan instrument image",
        domains=["pan", "soundfield", "masking"],
        instrument=group.instrument,
        problems=["crowded_center", "narrow_mix", "unclear_roles"],
        action_types=["pan_candidate"],
        limit=5,
    )

    group_audio = before_audio
    group_bands = before_metrics.get("band_energy") or {}
    low_mid = float(group_bands.get("low_mid", band_medians.get("low_mid", -24.0)))
    low_mid_excess = low_mid - float(band_medians.get("low_mid", low_mid))
    if low_mid_excess > 1.4:
        freq = 360.0 if group.instrument == "snare" else 300.0
        gain = -float(np.clip(0.8 + low_mid_excess * 0.22, 0.9, 2.2))
        for channel in group.channels:
            rendered_channels[channel] = _apply_stereo_peaking_eq(rendered_channels[channel], sr, freq, gain, 1.25)
            plans[channel].eq_bands.append((freq, gain, 1.25))
        actions.append({
            "type": "group_eq",
            "freq_hz": round(freq, 2),
            "gain_db": round(gain, 2),
            "q": 1.25,
            "problem": "summed_low_mid_buildup",
            "selected_rule_ids": _source_rule_ids(eq_matches),
        })
        group_audio = _sum_layer_group_audio(rendered_channels, group)

    level_metrics = _source_metrics(group_audio, sr, group.instrument)
    target = _layer_group_target_rms(base_target_rms_db, group.instrument)
    group_rms = float(level_metrics.get("rms_db", _audio_rms_db(group_audio)))
    max_group_boost = 4.0
    if group.instrument in {"overhead", "room", "hi_hat", "ride"}:
        max_group_boost = 1.5
    elif group.group_kind in {"stereo_pair", "support_stack"}:
        max_group_boost = 2.5
    group_gain = float(np.clip(target - group_rms, -8.0, max_group_boost))
    for channel in group.channels:
        _apply_layer_render_gain(
            rendered_channels,
            plans,
            channel,
            group_gain,
            "layer_group_summed_instrument_level",
        )
        plans[channel].trim_analysis.setdefault("layer_group", group.group_id)
    actions.append({
        "type": "group_level",
        "target_rms_db": round(target, 2),
        "before_group_rms_db": round(group_rms, 2),
        "gain_db": round(group_gain, 2),
        "max_boost_db": round(max_group_boost, 2),
        "selected_rule_ids": _source_rule_ids(level_matches),
    })

    if group.instrument in SOURCE_RULES_ONLY_ANCHORS:
        for channel in group.channels:
            plans[channel].pan = 0.0
            plans[channel].pan_notes.append({
                "mode": "layer_group_sum",
                "group_id": group.group_id,
                "decision": "group_anchor_centered",
                "selected_rule_ids": _source_rule_ids(pan_matches),
            })
        actions.append({
            "type": "group_position",
            "pan": 0.0,
            "reason": "summed_anchor_instrument_kept_center",
            "selected_rule_ids": _source_rule_ids(pan_matches),
        })
    else:
        after_position_audio = _sum_layer_group_audio(rendered_channels, group)
        left_db = _audio_rms_db(after_position_audio[:, 0]) if after_position_audio.ndim == 2 else _audio_rms_db(after_position_audio)
        right_db = _audio_rms_db(after_position_audio[:, 1]) if after_position_audio.ndim == 2 else left_db
        actions.append({
            "type": "group_position",
            "pan": "internal_layers",
            "left_rms_db": round(left_db, 2),
            "right_rms_db": round(right_db, 2),
            "center_offset_db": round(left_db - right_db, 2),
            "selected_rule_ids": _source_rule_ids(pan_matches),
        })

    after_audio = _sum_layer_group_audio(rendered_channels, group)
    after_metrics = _source_metrics(after_audio, sr, group.instrument)
    return actions, before_metrics, after_metrics


def apply_layer_group_mix_corrections(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    *,
    base_target_rms_db: float,
    band_medians: dict[str, float] | None = None,
    source_layer: SourceKnowledgeLayer | None = None,
    source_session_id: str = "",
) -> dict[str, Any]:
    """Treat multi-layer instruments as summed instruments after layer prep."""

    groups = build_layer_group_plans(plans)
    if not groups:
        return {"enabled": False, "reason": "no_layer_groups_detected", "groups": []}

    medians = dict(band_medians or _source_rules_only_band_medians(plans))
    group_reports: list[dict[str, Any]] = []
    for group in groups:
        if any(channel not in rendered_channels for channel in group.channels):
            continue
        internal_actions = _apply_layer_group_internal_balance(rendered_channels, plans, group)
        phase_actions = _apply_layer_group_phase(rendered_channels, plans, group, sr)
        group_actions, before_metrics, after_metrics = _apply_layer_group_sum_corrections(
            rendered_channels,
            plans,
            group,
            sr,
            base_target_rms_db,
            medians,
            source_layer,
        )
        report = {
            "group_id": group.group_id,
            "instrument": group.instrument,
            "group_kind": group.group_kind,
            "channels": group.channels,
            "files": [plans[channel].path.name for channel in group.channels],
            "roles": {str(channel): role for channel, role in group.roles.items()},
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "actions": [*internal_actions, *phase_actions, *group_actions],
        }
        for channel in group.channels:
            plans[channel].trim_analysis.setdefault("layer_group_reported", True)
            plans[channel].trim_analysis.setdefault("layer_group_id", group.group_id)
        group_reports.append(report)

    return {
        "enabled": bool(group_reports),
        "mode": "post_render_summed_layer_groups",
        "source_session_id": source_session_id,
        "groups": group_reports,
        "notes": [
            "Layer channels are first prepared individually, then summed and corrected as one instrument.",
            "Group level and image decisions use the summed layer audio, not isolated layer metrics.",
        ],
    }


def _source_decision_slug(*parts: Any) -> str:
    text = "_".join(str(part) for part in parts if str(part))
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")[:180] or "candidate"


def _fit_audio_to_len(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) < target_len:
        if audio.ndim == 1:
            return np.pad(audio, (0, target_len - len(audio))).astype(np.float32)
        return np.pad(audio, ((0, target_len - len(audio)), (0, 0))).astype(np.float32)
    return audio[:target_len].astype(np.float32, copy=False)


def trace_channel_source_candidates(
    layer: SourceKnowledgeLayer | None,
    *,
    session_id: str,
    channel: int,
    plan: ChannelPlan,
    target_len: int,
    sr: int,
) -> int:
    """Replay the offline channel strip only to log source-grounded candidates."""

    if layer is None or not layer.enabled:
        return 0
    logged = 0
    trace_plan = copy.deepcopy(plan)
    try:
        audio, file_sr = sf.read(trace_plan.path, dtype="float32", always_2d=False)
        if int(file_sr) != int(sr):
            return 0
    except Exception:
        return 0

    is_stereo_source = (
        isinstance(audio, np.ndarray)
        and audio.ndim == 2
        and audio.shape[1] == 2
        and trace_plan.instrument in {"overhead", "playback", "room"}
    )
    channel_name = f"{channel}:{trace_plan.path.name}"

    if is_stereo_source:
        stereo = _fit_audio_to_len(np.asarray(audio, dtype=np.float32), target_len)
        stereo = np.column_stack([
            declick_start(stereo[:, 0], sr, trace_plan.input_fade_ms),
            declick_start(stereo[:, 1], sr, trace_plan.input_fade_ms),
        ]).astype(np.float32)
        stereo = (stereo * db_to_amp(trace_plan.trim_db)).astype(np.float32)
        if trace_plan.phase_invert:
            stereo = (-stereo).astype(np.float32)
        if abs(trace_plan.delay_ms) > 1e-4:
            stereo = np.column_stack([
                delay_signal(stereo[:, 0], sr, trace_plan.delay_ms),
                delay_signal(stereo[:, 1], sr, trace_plan.delay_ms),
            ]).astype(np.float32)
        for channel_index in range(2):
            lane = stereo[:, channel_index]
            lane = highpass(lane, sr, trace_plan.hpf)
            if trace_plan.lpf > 0.0:
                lane = lowpass(lane, sr, trace_plan.lpf)
            stereo[:, channel_index] = lane.astype(np.float32)

        for band_index, (freq, gain, q) in enumerate(trace_plan.eq_bands, start=1):
            before = stereo.copy()
            for channel_index in range(2):
                stereo[:, channel_index] = peaking_eq(
                    stereo[:, channel_index],
                    sr,
                    freq,
                    gain,
                    q,
                )
            problems, label = _infer_eq_problem(trace_plan.instrument, freq, gain)
            matches = _safe_source_retrieve(
                layer,
                f"{trace_plan.instrument} {label} eq {freq:.0f}Hz {gain:.1f}dB",
                domains=["eq", "masking", "tone"],
                instrument=trace_plan.instrument,
                problems=problems,
                action_types=["eq_candidate"],
            )
            logged += int(_record_source_candidate(
                layer,
                session_id=session_id,
                decision_id=_source_decision_slug(session_id, channel, "eq", band_index),
                channel=channel_name,
                instrument=trace_plan.instrument,
                category="eq",
                problem=problems[0],
                matches=matches,
                action={
                    "action_type": "eq_candidate",
                    "band_index": band_index,
                    "freq_hz": round(float(freq), 3),
                    "gain_db": round(float(gain), 3),
                    "q": round(float(q), 3),
                },
                before_audio=before,
                after_audio=stereo,
                sr=sr,
                context={"stage": "channel_strip", "source_mode": "stereo_preserved"},
            ))

        before_comp = stereo.copy()
        for channel_index in range(2):
            stereo[:, channel_index] = compressor(
                stereo[:, channel_index],
                sr,
                threshold_db=trace_plan.comp_threshold_db,
                ratio=trace_plan.comp_ratio,
                attack_ms=trace_plan.comp_attack_ms,
                release_ms=trace_plan.comp_release_ms,
                makeup_db=0.0,
            )
        if trace_plan.comp_ratio > 1.01:
            matches = _safe_source_retrieve(
                layer,
                f"{trace_plan.instrument} compression level envelope density",
                domains=["dynamics", "compression"],
                instrument=trace_plan.instrument,
                problems=["unstable_level", "excessive_peaks", "weak_density"],
                action_types=["compressor_candidate"],
            )
            logged += int(_record_source_candidate(
                layer,
                session_id=session_id,
                decision_id=_source_decision_slug(session_id, channel, "comp"),
                channel=channel_name,
                instrument=trace_plan.instrument,
                category="compressor",
                problem="unstable_level",
                matches=matches,
                action={
                    "action_type": "compressor_candidate",
                    "threshold_db": round(float(trace_plan.comp_threshold_db), 3),
                    "ratio": round(float(trace_plan.comp_ratio), 3),
                    "attack_ms": round(float(trace_plan.comp_attack_ms), 3),
                    "release_ms": round(float(trace_plan.comp_release_ms), 3),
                },
                before_audio=before_comp,
                after_audio=stereo,
                sr=sr,
                context={"stage": "channel_strip", "source_mode": "stereo_preserved"},
            ))
        return logged

    mono = _fit_audio_to_len(mono_sum(audio), target_len)
    mono = declick_start(mono, sr, trace_plan.input_fade_ms)
    mono = (mono * db_to_amp(trace_plan.trim_db)).astype(np.float32)
    if trace_plan.phase_invert:
        mono = (-mono).astype(np.float32)
    mono = delay_signal(mono, sr, trace_plan.delay_ms)
    x = highpass(mono, sr, trace_plan.hpf)
    if trace_plan.lpf > 0.0:
        x = lowpass(x, sr, trace_plan.lpf)
    x, _ = apply_event_based_expander(x, sr, trace_plan)

    for band_index, (freq, gain, q) in enumerate(trace_plan.eq_bands, start=1):
        before = x.copy()
        x = peaking_eq(x, sr, freq, gain, q)
        problems, label = _infer_eq_problem(trace_plan.instrument, freq, gain)
        matches = _safe_source_retrieve(
            layer,
            f"{trace_plan.instrument} {label} eq {freq:.0f}Hz {gain:.1f}dB",
            domains=["eq", "masking", "tone"],
            instrument=trace_plan.instrument,
            problems=problems,
            action_types=["eq_candidate"],
        )
        logged += int(_record_source_candidate(
            layer,
            session_id=session_id,
            decision_id=_source_decision_slug(session_id, channel, "eq", band_index),
            channel=channel_name,
            instrument=trace_plan.instrument,
            category="eq",
            problem=problems[0],
            matches=matches,
            action={
                "action_type": "eq_candidate",
                "band_index": band_index,
                "freq_hz": round(float(freq), 3),
                "gain_db": round(float(gain), 3),
                "q": round(float(q), 3),
            },
            before_audio=before,
            after_audio=x,
            sr=sr,
            context={"stage": "channel_strip", "source_mode": "mono"},
        ))

    if trace_plan.comp_ratio > 1.01:
        before_comp = x.copy()
        after_comp = compressor(
            x,
            sr,
            threshold_db=trace_plan.comp_threshold_db,
            ratio=trace_plan.comp_ratio,
            attack_ms=trace_plan.comp_attack_ms,
            release_ms=trace_plan.comp_release_ms,
            makeup_db=0.0,
        )
        matches = _safe_source_retrieve(
            layer,
            f"{trace_plan.instrument} compression level envelope density",
            domains=["dynamics", "compression"],
            instrument=trace_plan.instrument,
            problems=["unstable_level", "excessive_peaks", "weak_density"],
            action_types=["compressor_candidate"],
        )
        logged += int(_record_source_candidate(
            layer,
            session_id=session_id,
            decision_id=_source_decision_slug(session_id, channel, "comp"),
            channel=channel_name,
            instrument=trace_plan.instrument,
            category="compressor",
            problem="unstable_level",
            matches=matches,
            action={
                "action_type": "compressor_candidate",
                "threshold_db": round(float(trace_plan.comp_threshold_db), 3),
                "ratio": round(float(trace_plan.comp_ratio), 3),
                "attack_ms": round(float(trace_plan.comp_attack_ms), 3),
                "release_ms": round(float(trace_plan.comp_release_ms), 3),
            },
            before_audio=before_comp,
            after_audio=after_comp,
            sr=sr,
            context={"stage": "channel_strip", "source_mode": "mono"},
        ))
        x = after_comp

    before_pan = pan_mono(x, 0.0) * db_to_amp(trace_plan.fader_db)
    after_pan = pan_mono(x, trace_plan.pan) * db_to_amp(trace_plan.fader_db)
    matches = _safe_source_retrieve(
        layer,
        f"{trace_plan.instrument} pan soundfield role center masking",
        domains=["pan", "soundfield", "balance"],
        instrument=trace_plan.instrument,
        problems=["crowded_center", "unclear_roles", "narrow_mix"],
        action_types=["pan_candidate"],
    )
    logged += int(_record_source_candidate(
        layer,
        session_id=session_id,
        decision_id=_source_decision_slug(session_id, channel, "pan"),
        channel=channel_name,
        instrument=trace_plan.instrument,
        category="pan",
        problem="crowded_center" if abs(trace_plan.pan) >= 0.05 else "unclear_roles",
        matches=matches,
        action={
            "action_type": "pan_candidate",
            "pan": round(float(trace_plan.pan), 4),
            "fader_db": round(float(trace_plan.fader_db), 3),
            "center_anchor": abs(trace_plan.pan) < 0.05,
        },
        before_audio=before_pan,
        after_audio=after_pan,
        sr=sr,
        context={"stage": "channel_strip", "source_mode": "mono"},
    ))
    return logged


def apply_event_based_expander(x: np.ndarray, sr: int, plan: ChannelPlan) -> tuple[np.ndarray, dict[str, Any]]:
    if not plan.expander_enabled or len(x) == 0:
        return x, {"enabled": False}

    activity = plan.event_activity or _event_activity_ranges(x, sr, plan.instrument)
    if not activity or not activity["ranges"]:
        return x, {
            "enabled": False,
            "reason": "no_event_ranges_detected",
        }

    frame = int(activity["frame"])
    hop = int(activity["hop"])
    starts = np.arange(0, max(1, len(x) - frame + 1), hop, dtype=np.int64)
    if len(starts) == 0:
        return x, {"enabled": False, "reason": "no_frames"}

    hold = int(plan.expander_hold_ms * sr / 1000.0)
    active = np.zeros(len(starts), dtype=bool)
    for start, end in activity["ranges"]:
        extended_end = min(len(x), end + hold)
        active |= (starts < extended_end) & ((starts + frame) > start)
    if not np.any(active):
        return x, {"enabled": False, "reason": "no_active_frames"}

    target_db = np.where(active, 0.0, -plan.expander_range_db).astype(np.float32)
    smoothed_db = _smooth_gain_db(
        target_db,
        sr,
        hop,
        attack_ms=plan.expander_close_ms,
        release_ms=plan.expander_open_ms,
    )

    sample_points = np.clip(starts + frame // 2, 0, len(x) - 1)
    full_points = np.concatenate(([0.0], sample_points.astype(np.float32), [float(len(x) - 1)]))
    full_gain = np.concatenate(([smoothed_db[0]], smoothed_db, [smoothed_db[-1]])).astype(np.float32)
    envelope_db = np.interp(np.arange(len(x), dtype=np.float32), full_points, full_gain).astype(np.float32)
    gain = np.power(10.0, envelope_db / 20.0, dtype=np.float32)
    out = (x * gain).astype(np.float32)

    inactive = ~active
    return out, {
        "enabled": True,
        "mode": "event_based_expander",
        "range_db": round(plan.expander_range_db, 2),
        "threshold_db": activity["threshold_db"],
        "open_ms": round(plan.expander_open_ms, 2),
        "close_ms": round(plan.expander_close_ms, 2),
        "hold_ms": round(plan.expander_hold_ms, 2),
        "active_ratio": round(float(np.mean(active)), 4),
        "active_sec": round(float(np.sum(active) * hop / sr), 3),
        "max_reduction_db": round(float(np.max(-smoothed_db)), 2),
        "mean_reduction_inactive_db": round(float(np.mean(-smoothed_db[inactive])), 2) if np.any(inactive) else 0.0,
        "mean_reduction_active_db": round(float(np.mean(-smoothed_db[active])), 2),
    }


def render_channel(path: Path, plan: ChannelPlan, target_len: int, sr: int) -> np.ndarray:
    audio, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{path.name}: expected {sr} Hz, got {file_sr} Hz")
    if isinstance(audio, np.ndarray) and audio.ndim == 2 and audio.shape[1] == 2 and plan.instrument in {"overhead", "playback", "room"}:
        stereo = audio.astype(np.float32)
        if len(stereo) < target_len:
            stereo = np.pad(stereo, ((0, target_len - len(stereo)), (0, 0)))
        stereo = stereo[:target_len]
        stereo = np.column_stack([
            declick_start(stereo[:, 0], sr, plan.input_fade_ms),
            declick_start(stereo[:, 1], sr, plan.input_fade_ms),
        ]).astype(np.float32)
        stereo = (stereo * db_to_amp(plan.trim_db)).astype(np.float32)
        if plan.phase_invert:
            stereo = (-stereo).astype(np.float32)
        if abs(plan.delay_ms) > 1e-4:
            stereo = np.column_stack([
                delay_signal(stereo[:, 0], sr, plan.delay_ms),
                delay_signal(stereo[:, 1], sr, plan.delay_ms),
            ]).astype(np.float32)
        for channel_index in range(2):
            lane = stereo[:, channel_index]
            lane = highpass(lane, sr, plan.hpf)
            if plan.lpf > 0.0:
                lane = lowpass(lane, sr, plan.lpf)
            for freq, gain, q in plan.eq_bands:
                lane = peaking_eq(lane, sr, freq, gain, q)
            lane = compressor(
                lane,
                sr,
                threshold_db=plan.comp_threshold_db,
                ratio=plan.comp_ratio,
                attack_ms=plan.comp_attack_ms,
                release_ms=plan.comp_release_ms,
                makeup_db=0.0,
            )
            stereo[:, channel_index] = lane.astype(np.float32)
        plan.expander_report = {"enabled": False, "reason": "stereo_source_preserved"}
        return (stereo * db_to_amp(plan.fader_db)).astype(np.float32)
    mono = mono_sum(audio)
    if len(mono) < target_len:
        mono = np.pad(mono, (0, target_len - len(mono)))
    mono = mono[:target_len]
    mono = declick_start(mono, sr, plan.input_fade_ms)
    mono = mono * db_to_amp(plan.trim_db)
    if plan.phase_invert:
        mono = -mono
    mono = delay_signal(mono, sr, plan.delay_ms)
    x = mono
    x = highpass(x, sr, plan.hpf)
    if plan.lpf > 0.0:
        x = lowpass(x, sr, plan.lpf)
    x, plan.expander_report = apply_event_based_expander(x, sr, plan)
    for freq, gain, q in plan.eq_bands:
        x = peaking_eq(x, sr, freq, gain, q)
    x = compressor(
        x,
        sr,
        threshold_db=plan.comp_threshold_db,
        ratio=plan.comp_ratio,
        attack_ms=plan.comp_attack_ms,
        release_ms=plan.comp_release_ms,
        makeup_db=0.0,
    )
    stereo = pan_mono(x, plan.pan)
    return (stereo * db_to_amp(plan.fader_db)).astype(np.float32)


def _render_frequency_window_preview(
    plan: ChannelPlan,
    sr: int,
    preview_sec: float = FREQUENCY_WINDOW_PREVIEW_SEC,
) -> np.ndarray:
    """Render a bounded channel preview for broad frequency-window decisions."""

    audio, file_sr = sf.read(plan.path, dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{plan.path.name}: expected {sr} Hz, got {file_sr} Hz")

    preview_len = max(1024, int(float(preview_sec) * sr))
    if (
        isinstance(audio, np.ndarray)
        and audio.ndim == 2
        and audio.shape[1] == 2
        and plan.instrument in {"overhead", "playback", "room"}
    ):
        stereo_source = audio.astype(np.float32, copy=False)
        mono_reference = mono_sum(stereo_source)
        start = _active_segment_start(
            mono_reference,
            sr,
            window_sec=float(preview_sec),
        )
        stereo = stereo_source[start:start + preview_len]
        if len(stereo) < preview_len:
            stereo = np.pad(stereo, ((0, preview_len - len(stereo)), (0, 0)))
        stereo = np.column_stack([
            declick_start(stereo[:, 0], sr, plan.input_fade_ms),
            declick_start(stereo[:, 1], sr, plan.input_fade_ms),
        ]).astype(np.float32)
        stereo = (stereo * db_to_amp(plan.trim_db)).astype(np.float32)
        if plan.phase_invert:
            stereo = (-stereo).astype(np.float32)
        if abs(plan.delay_ms) > 1e-4:
            stereo = np.column_stack([
                delay_signal(stereo[:, 0], sr, plan.delay_ms),
                delay_signal(stereo[:, 1], sr, plan.delay_ms),
            ]).astype(np.float32)
        for channel_index in range(2):
            lane = stereo[:, channel_index]
            lane = highpass(lane, sr, plan.hpf)
            if plan.lpf > 0.0:
                lane = lowpass(lane, sr, plan.lpf)
            for freq, gain, q in plan.eq_bands:
                lane = peaking_eq(lane, sr, freq, gain, q)
            lane = compressor(
                lane,
                sr,
                threshold_db=plan.comp_threshold_db,
                ratio=plan.comp_ratio,
                attack_ms=plan.comp_attack_ms,
                release_ms=plan.comp_release_ms,
                makeup_db=0.0,
            )
            stereo[:, channel_index] = lane.astype(np.float32)
        return (stereo * db_to_amp(plan.fader_db)).astype(np.float32)

    mono = mono_sum(audio)
    block = _analysis_block(
        mono,
        sr,
        window_sec=float(preview_sec),
    )
    if len(block) < preview_len:
        block = np.pad(block, (0, preview_len - len(block)))
    else:
        block = block[:preview_len]
    x = declick_start(block.astype(np.float32), sr, plan.input_fade_ms)
    x = x * db_to_amp(plan.trim_db)
    if plan.phase_invert:
        x = -x
    x = delay_signal(x, sr, plan.delay_ms)
    x = highpass(x, sr, plan.hpf)
    if plan.lpf > 0.0:
        x = lowpass(x, sr, plan.lpf)
    for freq, gain, q in plan.eq_bands:
        x = peaking_eq(x, sr, freq, gain, q)
    x = compressor(
        x,
        sr,
        threshold_db=plan.comp_threshold_db,
        ratio=plan.comp_ratio,
        attack_ms=plan.comp_attack_ms,
        release_ms=plan.comp_release_ms,
        makeup_db=0.0,
    )
    stereo = pan_mono(x, plan.pan)
    return (stereo * db_to_amp(plan.fader_db)).astype(np.float32)


def _round_render_value(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _event_activity_signature(activity: dict[str, Any]) -> tuple[Any, ...]:
    ranges = activity.get("ranges") if isinstance(activity, dict) else None
    if not ranges:
        return ()
    return (
        int(activity.get("frame", 0) or 0),
        int(activity.get("hop", 0) or 0),
        _round_render_value(activity.get("threshold_db", 0.0), 3),
        tuple((int(start), int(end)) for start, end in ranges[:128]),
        int(activity.get("active_samples", 0) or 0),
    )


def _plan_render_signature(plan: ChannelPlan, sr: int, target_len: int, mode: str) -> tuple[Any, ...]:
    try:
        stat = plan.path.stat()
        file_signature = (
            str(plan.path.resolve()),
            int(stat.st_mtime_ns),
            int(stat.st_size),
        )
    except OSError:
        file_signature = (str(plan.path), 0, 0)
    return (
        mode,
        file_signature,
        int(sr),
        int(target_len),
        plan.instrument,
        bool(plan.muted),
        _round_render_value(plan.pan),
        _round_render_value(plan.hpf),
        _round_render_value(plan.lpf),
        _round_render_value(plan.trim_db),
        _round_render_value(plan.fader_db),
        bool(plan.phase_invert),
        _round_render_value(plan.delay_ms),
        _round_render_value(plan.input_fade_ms),
        tuple(
            (
                _round_render_value(freq, 3),
                _round_render_value(gain, 4),
                _round_render_value(q, 4),
            )
            for freq, gain, q in plan.eq_bands
        ),
        _round_render_value(plan.comp_threshold_db),
        _round_render_value(plan.comp_ratio),
        _round_render_value(plan.comp_attack_ms),
        _round_render_value(plan.comp_release_ms),
        bool(plan.expander_enabled),
        _round_render_value(plan.expander_range_db),
        _round_render_value(plan.expander_open_ms),
        _round_render_value(plan.expander_close_ms),
        _round_render_value(plan.expander_hold_ms),
        _round_render_value(plan.expander_threshold_db or 0.0),
        _event_activity_signature(plan.event_activity),
    )


class OfflineRenderCache:
    """Small LRU cache for expensive offline channel renders."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_bytes: int = int(DEFAULT_RENDER_CACHE_MAX_MB * 1024 * 1024),
    ):
        self.enabled = bool(enabled)
        self.max_bytes = max(0, int(max_bytes))
        self._items: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_or_render(self, key: tuple[Any, ...], render_fn) -> np.ndarray:
        if not self.enabled or self.max_bytes <= 0:
            self.misses += 1
            return render_fn()
        cached = self._items.get(key)
        if cached is not None:
            self.hits += 1
            self._items.move_to_end(key)
            return cached.copy()

        self.misses += 1
        rendered = np.asarray(render_fn(), dtype=np.float32)
        self._store(key, rendered)
        return rendered.copy()

    def _store(self, key: tuple[Any, ...], value: np.ndarray) -> None:
        if key in self._items:
            old = self._items.pop(key)
            self._bytes -= int(old.nbytes)
        self._items[key] = value.copy()
        self._bytes += int(value.nbytes)
        self._items.move_to_end(key)
        while self._bytes > self.max_bytes and self._items:
            _old_key, old_value = self._items.popitem(last=False)
            self._bytes -= int(old_value.nbytes)
            self.evictions += 1

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
            "items": int(len(self._items)),
            "bytes": int(self._bytes),
            "max_bytes": int(self.max_bytes),
        }


def render_channel_cached(
    channel: int,
    plan: ChannelPlan,
    target_len: int,
    sr: int,
    render_cache: OfflineRenderCache | None = None,
) -> np.ndarray:
    if render_cache is None:
        return render_channel(plan.path, plan, target_len, sr)
    key = ("full", int(channel), _plan_render_signature(plan, sr, target_len, "full"))
    return render_cache.get_or_render(key, lambda: render_channel(plan.path, plan, target_len, sr))


def render_channel_preview_cached(
    channel: int,
    plan: ChannelPlan,
    sr: int,
    *,
    preview_sec: float,
    render_cache: OfflineRenderCache | None = None,
) -> np.ndarray:
    target_len = max(1024, int(float(preview_sec) * sr))
    if render_cache is None:
        if not plan.path.exists():
            return render_channel(plan.path, plan, target_len, sr)
        return _render_frequency_window_preview(plan, sr, preview_sec=preview_sec)
    key = (
        "preview",
        int(channel),
        _round_render_value(preview_sec, 3),
        _plan_render_signature(plan, sr, target_len, "preview"),
    )
    return render_cache.get_or_render(
        key,
        lambda: (
            render_channel(plan.path, plan, target_len, sr)
            if not plan.path.exists()
            else _render_frequency_window_preview(plan, sr, preview_sec=preview_sec)
        ),
    )


def apply_live_channel_peak_headroom(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    channel_peak_ceiling_db: float = -3.0,
    render_cache: OfflineRenderCache | None = None,
) -> dict[str, Any]:
    """Use static channel fader trims for no-master-limiter live headroom."""
    adjusted = []
    for channel, plan in plans.items():
        if plan.muted:
            continue
        rendered = render_channel_cached(channel, plan, target_len, sr, render_cache)
        peak_db = amp_to_db(float(np.max(np.abs(rendered))) if len(rendered) else 0.0)
        reduction_db = min(0.0, channel_peak_ceiling_db - peak_db)
        if reduction_db < -0.05:
            plan.fader_db = float(np.clip(plan.fader_db + reduction_db, -100.0, 10.0))
            adjusted.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "rendered_peak_before_dbfs": round(peak_db, 2),
                "fader_reduction_db": round(reduction_db, 2),
                "fader_after_db": round(plan.fader_db, 2),
            })
    return {
        "enabled": True,
        "channel_peak_ceiling_dbfs": round(channel_peak_ceiling_db, 2),
        "adjusted_channels": adjusted,
    }


def apply_bass_drum_push(plans: dict[int, ChannelPlan], boost_db: float) -> dict[str, Any]:
    """Push bass and drum stems for a fuller low-end and punchy kit picture."""
    if boost_db <= 0.0:
        return {"enabled": False}

    target_instruments = {
        "bass_guitar",
        "kick",
        "snare",
        "rack_tom",
        "floor_tom",
        "hi_hat",
        "ride",
    }
    changes = []
    for channel, plan in plans.items():
        if plan.muted or plan.instrument not in target_instruments:
            continue
        before = plan.fader_db
        after = float(np.clip(before + boost_db, -100.0, 10.0))
        if after == before:
            continue
        plan.fader_db = after
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "fader_before_db": round(before, 2),
            "fader_after_db": round(after, 2),
            "boost_db": round(boost_db, 2),
        })

    return {
        "enabled": bool(changes),
        "boost_db": round(boost_db, 2),
        "affected_channels": len(changes),
        "changes": changes,
    }


def apply_kick_presence_boost(plans: dict[int, ChannelPlan], boost_db: float) -> dict[str, Any]:
    """Raise only the kick stem and add subtle low-end shaping for stronger punch."""
    if boost_db <= 0.0:
        return {"enabled": False}

    changes = []
    for channel, plan in plans.items():
        if plan.muted or plan.instrument != "kick":
            continue

        fader_before = plan.fader_db
        fader_after = float(np.clip(fader_before + boost_db, -100.0, 10.0))
        if fader_after != fader_before:
            plan.fader_db = fader_after
            changes.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "fader_before_db": round(fader_before, 2),
                "fader_after_db": round(fader_after, 2),
                "boost_db": round(boost_db, 2),
            })

        eq_before = list(plan.eq_bands)
        if boost_db >= 2.0:
            # Emphasize body + beater click for clearer kick translation on big systems.
            plan.eq_bands.append((65.0, 1.2, 1.0))
            plan.eq_bands.append((3000.0, 0.9, 1.8))
            changes.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "change_type": "eq_shape_added",
                "eq_before_count": len(eq_before),
                "eq_after_count": len(plan.eq_bands),
            })

    return {
        "enabled": bool(changes),
        "boost_db": round(boost_db, 2),
        "changes": changes,
    }


def _band_rms_db(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    mono = mono_sum(audio)
    block = _analysis_block(mono, sr, window_sec=BAND_ANALYSIS_WINDOW_SEC)
    if len(block) < 256:
        return -100.0
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    idx = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(idx):
        return -100.0
    band_rms = float(np.sqrt(np.mean(np.square(spec[idx]))) + 1e-12)
    return amp_to_db(band_rms)


def _band_power(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    mono = mono_sum(audio)
    block = _analysis_block(mono, sr, window_sec=BAND_ANALYSIS_WINDOW_SEC)
    if len(block) < 256:
        return 1e-12
    windowed = block * np.hanning(len(block))
    spec = np.abs(np.fft.rfft(windowed)) + 1e-12
    power = np.square(spec)
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)
    idx = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(idx):
        return 1e-12
    return float(np.sum(power[idx]) + 1e-12)


def _mirror_eq_band_levels(
    audio: np.ndarray,
    sr: int,
    centers_hz: tuple[float, ...] = MIRROR_EQ_CENTERS_HZ,
    *,
    window_sec: float = 24.0,
) -> dict[float, float]:
    mono = mono_sum(normalize_audio_shape(audio))
    block = _analysis_block(mono, sr, window_sec=window_sec)
    if len(block) < 256:
        return {float(center): -100.0 for center in centers_hz}

    windowed = block.astype(np.float32) * np.hanning(len(block)).astype(np.float32)
    power = np.square(np.abs(np.fft.rfft(windowed)).astype(np.float64)) + 1e-12
    freqs = np.fft.rfftfreq(len(windowed), 1.0 / sr)

    levels: dict[float, float] = {}
    for center in centers_hz:
        width = math.sqrt(2.0)
        low_hz = max(20.0, float(center) / width)
        high_hz = min(sr * 0.49, float(center) * width)
        mask = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(mask):
            levels[float(center)] = -100.0
            continue
        levels[float(center)] = 10.0 * math.log10(float(np.mean(power[mask])) + 1e-12)
    return levels


def _mirror_eq_normalized_profile(levels: dict[float, float]) -> tuple[dict[float, float], float]:
    anchors = [
        float(levels[center])
        for center in (500.0, 1000.0, 2000.0)
        if center in levels and np.isfinite(levels[center]) and levels[center] > -99.0
    ]
    anchor_db = float(np.mean(anchors)) if anchors else 0.0
    return {float(center): float(value - anchor_db) for center, value in levels.items()}, anchor_db


def _mirror_eq_shape_metrics(levels: dict[float, float]) -> dict[str, float]:
    lowmid = float(np.mean([levels.get(250.0, -100.0), levels.get(350.0, -100.0), levels.get(500.0, -100.0)]))
    presence = float(np.mean([levels.get(2000.0, -100.0), levels.get(4000.0, -100.0)]))
    top = float(np.mean([levels.get(4000.0, -100.0), levels.get(8000.0, -100.0)]))
    return {
        "lowmid_250_500_minus_presence_2_4k_db": float(lowmid - presence),
        "top_4_8k_minus_lowmid_250_500_db": float(top - lowmid),
    }


def _mirror_eq_target_bounds(center_hz: float) -> tuple[float, float]:
    center = float(center_hz)
    if center <= 80.0:
        return 10.0, 20.0
    if center <= 160.0:
        return 7.0, 14.0
    if center <= 280.0:
        return 4.0, 9.5
    if center <= 420.0:
        return 3.0, 8.0
    if center <= 700.0:
        return 1.0, 6.0
    if center <= 1500.0:
        return -2.5, 2.5
    if center <= 3000.0:
        return -5.0, 1.0
    if center <= 6000.0:
        return -10.0, -2.0
    return -16.0, -6.5


def _mirror_eq_q(center_hz: float) -> float:
    center = float(center_hz)
    if center <= 160.0:
        return 0.85
    if center <= 500.0:
        return 0.95
    if center <= 2000.0:
        return 1.15
    return 1.3


def apply_reference_mirror_master_eq(
    mix: np.ndarray,
    sr: int,
    reference_audio: np.ndarray,
    reference_sr: int,
    *,
    centers_hz: tuple[float, ...] = MIRROR_EQ_CENTERS_HZ,
    strength: float = 0.42,
    lowmid_floor_db: float = 7.5,
    lowmid_ceiling_db: float = 12.0,
    white_noise_top_floor_db: float = -7.0,
    max_boost_db: float = 1.8,
    max_cut_db: float = 2.4,
    window_sec: float = 24.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply conservative reference mirror EQ on the master bus.

    This intentionally does not match the reference literally. It mirrors the
    smoothed reference/current residual, then clamps the result into a broad
    mix-safe corridor so dark references do not overfill low-mid and bright
    references do not push the mix toward a white-noise tilt.
    """

    if mix.size == 0:
        return np.asarray(mix, dtype=np.float32), {"enabled": False, "reason": "empty_mix"}

    prepared_reference = normalize_audio_shape(reference_audio)
    if int(reference_sr) != int(sr):
        prepared_reference = resample_audio(prepared_reference, int(reference_sr), int(sr))

    before_levels = _mirror_eq_band_levels(mix, sr, centers_hz, window_sec=window_sec)
    reference_levels = _mirror_eq_band_levels(prepared_reference, sr, centers_hz, window_sec=window_sec)
    before_norm, before_anchor = _mirror_eq_normalized_profile(before_levels)
    reference_norm, reference_anchor = _mirror_eq_normalized_profile(reference_levels)
    before_shape = _mirror_eq_shape_metrics(before_norm)

    lowmid_metric = before_shape["lowmid_250_500_minus_presence_2_4k_db"]
    top_metric = before_shape["top_4_8k_minus_lowmid_250_500_db"]
    pending_corrections: list[dict[str, Any]] = []

    for center in centers_hz:
        center = float(center)
        current = float(before_norm.get(center, 0.0))
        reference = float(reference_norm.get(center, current))
        lower, upper = _mirror_eq_target_bounds(center)
        mirror_target = current + (reference - current) * float(np.clip(strength, 0.0, 1.0))
        target = float(np.clip(mirror_target, lower, upper))

        # If the overall low-mid is underfilled, deliberately restore a little
        # 250-500 Hz even when the reference residual is ambiguous. If it is
        # already full, do not allow the dark-reference mirror to add more.
        if 180.0 <= center <= 500.0:
            if lowmid_metric < lowmid_floor_db:
                needed = (lowmid_floor_db - lowmid_metric) * 0.45
                target = max(target, min(upper, current + needed))
            elif lowmid_metric > lowmid_ceiling_db:
                target = min(target, max(lower, current - (lowmid_metric - lowmid_ceiling_db) * 0.35))

        # The old pink-noise pass failed here: top end moved above low-mid.
        # Only permit high boosts when the mix is not already bright by shape.
        if center >= 4000.0 and top_metric > white_noise_top_floor_db:
            target = min(target, current)

        raw_gain = target - current
        band_max_boost = max_boost_db
        band_max_cut = max_cut_db
        if center <= 80.0:
            band_max_boost = 0.5
        elif 180.0 <= center <= 500.0:
            band_max_boost = max(max_boost_db, 2.2)
            band_max_cut = min(max_cut_db, 1.8)
        elif center >= 4000.0:
            band_max_boost = min(max_boost_db, 0.9)

        gain_db = float(np.clip(raw_gain, -band_max_cut, band_max_boost))
        if abs(gain_db) < 0.18:
            continue

        q = _mirror_eq_q(center)
        pending_corrections.append({
            "frequency_hz": round(center, 2),
            "requested_gain_db": round(gain_db, 2),
            "q": round(q, 2),
            "current_normalized_db": round(current, 2),
            "reference_normalized_db": round(reference, 2),
            "mirror_target_normalized_db": round(mirror_target, 2),
            "safe_target_normalized_db": round(target, 2),
            "bounds_db": [round(lower, 2), round(upper, 2)],
        })

    def render_scaled(scale: float) -> tuple[np.ndarray, list[dict[str, Any]], dict[float, float], float, dict[str, float]]:
        rendered = np.asarray(mix, dtype=np.float32).copy()
        scaled_corrections: list[dict[str, Any]] = []
        for item in pending_corrections:
            gain_db = float(item["requested_gain_db"]) * float(scale)
            if abs(gain_db) < 0.08:
                continue
            center = float(item["frequency_hz"])
            q = float(item["q"])
            for channel in range(rendered.shape[1] if rendered.ndim == 2 else 1):
                if rendered.ndim == 1:
                    rendered = peaking_eq(rendered, sr, center, gain_db, q)
                    break
                rendered[:, channel] = peaking_eq(rendered[:, channel], sr, center, gain_db, q)
            scaled_item = dict(item)
            scaled_item["gain_db"] = round(gain_db, 2)
            scaled_item["scale"] = round(float(scale), 3)
            scaled_corrections.append(scaled_item)

        scaled_levels = _mirror_eq_band_levels(rendered, sr, centers_hz, window_sec=window_sec)
        scaled_norm, scaled_anchor = _mirror_eq_normalized_profile(scaled_levels)
        scaled_shape = _mirror_eq_shape_metrics(scaled_norm)
        return rendered.astype(np.float32), scaled_corrections, scaled_norm, scaled_anchor, scaled_shape

    def scale_score(shape: dict[str, float]) -> float:
        target_lowmid = 0.5 * (float(lowmid_floor_db) + float(lowmid_ceiling_db))
        lowmid = float(shape["lowmid_250_500_minus_presence_2_4k_db"])
        top = float(shape["top_4_8k_minus_lowmid_250_500_db"])
        score = -abs(lowmid - target_lowmid)
        score -= max(0.0, lowmid - float(lowmid_ceiling_db) - 0.6) * 3.5
        score -= max(0.0, float(lowmid_floor_db) - lowmid) * 1.8
        score -= max(0.0, top - float(white_noise_top_floor_db)) * 3.0
        return float(score)

    scale_candidates = (1.0, 0.75, 0.55, 0.4, 0.3, 0.22, 0.15)
    best_render = None
    for scale in scale_candidates:
        candidate = render_scaled(scale)
        candidate_score = scale_score(candidate[4])
        if best_render is None or candidate_score > best_render[0]:
            best_render = (candidate_score, scale, *candidate)

    if best_render is None:
        out = np.asarray(mix, dtype=np.float32).copy()
        corrections: list[dict[str, Any]] = []
        after_norm = dict(before_norm)
        after_anchor = before_anchor
        after_shape = dict(before_shape)
        selected_scale = 0.0
    else:
        _score, selected_scale, out, corrections, after_norm, after_anchor, after_shape = best_render

    after_levels = _mirror_eq_band_levels(out, sr, centers_hz, window_sec=window_sec)
    after_norm, after_anchor = _mirror_eq_normalized_profile(after_levels)
    after_shape = _mirror_eq_shape_metrics(after_norm)

    return out.astype(np.float32), {
        "enabled": bool(corrections),
        "mode": "bounded_reference_mirror_eq",
        "reference_sample_rate": int(reference_sr),
        "analysis_window_sec": round(float(window_sec), 2),
        "strength": round(float(strength), 3),
        "lowmid_floor_db": round(float(lowmid_floor_db), 2),
        "lowmid_ceiling_db": round(float(lowmid_ceiling_db), 2),
        "white_noise_top_floor_db": round(float(white_noise_top_floor_db), 2),
        "max_boost_db": round(float(max_boost_db), 2),
        "max_cut_db": round(float(max_cut_db), 2),
        "selected_scale": round(float(selected_scale), 3),
        "before": {
            "anchor_db": round(before_anchor, 2),
            "normalized_db": {str(int(center)): round(value, 2) for center, value in before_norm.items()},
            **{key: round(value, 2) for key, value in before_shape.items()},
        },
        "reference": {
            "anchor_db": round(reference_anchor, 2),
            "normalized_db": {str(int(center)): round(value, 2) for center, value in reference_norm.items()},
        },
        "after": {
            "anchor_db": round(after_anchor, 2),
            "normalized_db": {str(int(center)): round(value, 2) for center, value in after_norm.items()},
            **{key: round(value, 2) for key, value in after_shape.items()},
        },
        "corrections": corrections,
        "notes": [
            "Mirror EQ uses normalized smoothed band deltas, not absolute loudness.",
            "Dark references are clamped so low-mid is restored without copying their whole 180-500 Hz buildup.",
            "High-frequency boosts are blocked when the mix shape is already too close to a white-noise tilt.",
        ],
    }


def _sampled_band_dynamic_range_db(
    audio: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    *,
    window_sec: float = 0.35,
    segments: int = 8,
) -> float:
    mono = mono_sum(audio)
    if len(mono) < 1024:
        return 0.0
    starts = _active_segment_starts(mono, sr, window_sec=window_sec, count=segments)
    if not starts:
        starts = [0]
    values = []
    block_len = max(1024, int(window_sec * sr))
    for start in starts:
        end = min(len(mono), start + block_len)
        block = mono[start:end]
        if len(block) < 512:
            continue
        values.append(_band_rms_db(block, sr, low_hz, high_hz))
    if len(values) < 2:
        return 0.0
    return float(np.percentile(values, 90.0) - np.percentile(values, 20.0))


def _frequency_window_family(plan: ChannelPlan) -> str:
    instrument = plan.instrument
    if instrument == "lead_vocal":
        return "lead_vocal"
    if instrument == "backing_vocal":
        return "backing_vocal"
    if instrument in BASS_INSTRUMENTS:
        return "bass"
    if instrument == "kick":
        return "kick"
    if instrument in CYMBAL_INSTRUMENTS:
        return "cymbals"
    if instrument in DRUM_INSTRUMENTS:
        return "drums"
    return "music"


def _frequency_window_snapshot(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
) -> dict[str, Any]:
    if not rendered_channels:
        return {"windows": [], "by_id": {}}

    spectrum_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    valid_channels: list[tuple[int, ChannelPlan, np.ndarray]] = []
    family_audio: dict[str, np.ndarray] = {}
    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        valid_channels.append((channel, plan, audio))
        family = _frequency_window_family(plan)
        if family not in family_audio:
            family_audio[family] = np.zeros_like(audio)
        family_audio[family] += audio

    def _cached_band_metrics(audio: np.ndarray, low_hz: float, high_hz: float) -> tuple[float, float]:
        key = id(audio)
        cached = spectrum_cache.get(key)
        if cached is None:
            mono = mono_sum(audio)
            block = _analysis_block(mono, sr, window_sec=BAND_ANALYSIS_WINDOW_SEC)
            if len(block) < 256:
                cached = (
                    np.array([], dtype=np.float64),
                    np.array([], dtype=np.float64),
                )
            else:
                windowed = block * np.hanning(len(block))
                spec = np.abs(np.fft.rfft(windowed)) + 1e-12
                cached = (
                    np.fft.rfftfreq(len(windowed), 1.0 / sr),
                    np.square(spec),
                )
            spectrum_cache[key] = cached

        freqs, power = cached
        if freqs.size == 0:
            return 1e-12, -100.0
        idx = (freqs >= low_hz) & (freqs < high_hz)
        if not np.any(idx):
            return 1e-12, -100.0
        band_power = float(np.sum(power[idx]) + 1e-12)
        band_rms = float(np.sqrt(np.mean(power[idx])) + 1e-12)
        return band_power, amp_to_db(band_rms)

    windows: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for config in FREQUENCY_WINDOW_DEFINITIONS:
        low_hz = float(config["low_hz"])
        high_hz = float(config["high_hz"])
        total_power = sum(
            _cached_band_metrics(audio, low_hz, high_hz)[0]
            for _channel, _plan, audio in valid_channels
        )
        total_power = max(total_power, 1e-12)

        top_channels = []
        for channel, plan, audio in valid_channels:
            power, band_db = _cached_band_metrics(audio, low_hz, high_hz)
            top_channels.append({
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "family": _frequency_window_family(plan),
                "band_db": round(band_db, 2),
                "share": round(float(power / total_power), 4),
                "dynamic_range_db": round(_sampled_band_dynamic_range_db(audio, sr, low_hz, high_hz), 2),
            })
        top_channels.sort(key=lambda item: item["share"], reverse=True)

        family_metrics = []
        family_shares: dict[str, float] = {}
        for family, audio in family_audio.items():
            power, band_db = _cached_band_metrics(audio, low_hz, high_hz)
            share = float(power / total_power)
            family_shares[family] = round(share, 4)
            family_metrics.append({
                "family": family,
                "band_db": round(band_db, 2),
                "share": round(share, 4),
                "dynamic_range_db": round(_sampled_band_dynamic_range_db(audio, sr, low_hz, high_hz), 2),
            })
        family_metrics.sort(key=lambda item: item["share"], reverse=True)

        dominant_family = family_metrics[0]["family"] if family_metrics else ""
        runner_up_family = family_metrics[1]["family"] if len(family_metrics) > 1 else ""
        report = {
            "id": config["id"],
            "label": config["label"],
            "range_hz": [round(low_hz, 1), round(high_hz, 1)],
            "focus": config["focus"],
            "action_mode": config["action_mode"],
            "family_shares": family_shares,
            "dominant_family": dominant_family,
            "runner_up_family": runner_up_family,
            "families": family_metrics[:6],
            "channels": top_channels[:8],
        }
        windows.append(report)
        by_id[config["id"]] = report

    return {
        "windows": windows,
        "by_id": by_id,
    }


def _stem_roles_for_plan(plan: ChannelPlan) -> list[str]:
    classification = classification_from_legacy_preset(
        _legacy_preset_for_instrument(plan.instrument),
        channel_name=plan.name,
        confidence=1.0,
        match_type="offline_measurement",
    )
    stems = [stem.value for stem in classification.stem_roles if stem.value != "MASTER"]
    if not stems:
        stems = ["LEAD"] if plan.instrument == "lead_vocal" else ["UNKNOWN"]
    return stems


def _build_rendered_stem_groups(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
) -> tuple[dict[str, np.ndarray], dict[str, list[int]], dict[str, list[str]]]:
    stem_audio: dict[str, np.ndarray] = {}
    stem_channels: dict[str, list[int]] = {}
    stem_files: dict[str, list[str]] = {}

    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        for stem in _stem_roles_for_plan(plan):
            if stem not in stem_audio:
                stem_audio[stem] = np.zeros_like(audio)
                stem_channels[stem] = []
                stem_files[stem] = []
            stem_audio[stem] += audio
            stem_channels[stem].append(channel)
            stem_files[stem].append(plan.path.name)

    if rendered_channels:
        first_audio = next(iter(rendered_channels.values()))
        master_audio = np.zeros_like(first_audio)
        for audio in rendered_channels.values():
            master_audio += audio
        stem_audio["MASTER"] = master_audio
        stem_channels["MASTER"] = sorted(rendered_channels)
        stem_files["MASTER"] = [plans[channel].path.name for channel in sorted(rendered_channels) if channel in plans]

    return stem_audio, stem_channels, stem_files


def _activity_window_audio(
    audio: np.ndarray,
    ranges: list[tuple[int, int]] | None,
    sr: int,
    *,
    max_total_sec: float = 4.0,
) -> np.ndarray:
    mono = mono_sum(audio)
    if len(mono) == 0:
        return mono.astype(np.float32)
    if not ranges:
        return _analysis_block(mono, sr, window_sec=min(max_total_sec, 3.0))

    total_limit = max(1024, int(max_total_sec * sr))
    parts: list[np.ndarray] = []
    collected = 0
    for start, end in ranges:
        start_i = max(0, int(start))
        end_i = min(len(mono), int(end))
        if end_i <= start_i:
            continue
        segment = mono[start_i:end_i]
        if len(segment) == 0:
            continue
        remaining = total_limit - collected
        if remaining <= 0:
            break
        if len(segment) > remaining:
            segment = segment[:remaining]
        parts.append(segment.astype(np.float32))
        collected += len(segment)
    if not parts:
        return _analysis_block(mono, sr, window_sec=min(max_total_sec, 3.0))
    return np.concatenate(parts, axis=0).astype(np.float32)


def _ltas_tilt_profile(
    audio: np.ndarray,
    sr: int,
    *,
    compensation_db_per_octave: float = 4.5,
    window_sec: float = 3.0,
    segments: int = 6,
    smoothing_fraction: int = 3,
) -> dict[str, Any]:
    mono = mono_sum(audio)
    if len(mono) == 0:
        return {
            "compensation_db_per_octave": compensation_db_per_octave,
            "window_sec": window_sec,
            "segments": 0,
            "smoothing": f"1/{smoothing_fraction} octave",
            "regions_db": {},
            "weight_60_120_vs_plateau_db": 0.0,
            "sub_35_60_vs_plateau_db": 0.0,
            "high_4k5_12k_vs_plateau_db": 0.0,
            "plateau_spread_db": 0.0,
            "curve": [],
        }

    window = max(2048, int(window_sec * sr))
    starts = _active_segment_starts(mono, sr, window_sec=window_sec, count=segments)
    if not starts:
        starts = [0]

    spectra = []
    for start in starts:
        end = min(len(mono), start + window)
        block = mono[start:end]
        if len(block) < window:
            block = np.pad(block, (0, window - len(block)))
        block = block.astype(np.float32) * np.hanning(len(block)).astype(np.float32)
        spec = np.abs(np.fft.rfft(block)) + 1e-12
        spectra.append(np.square(spec))
    avg_power = np.mean(np.stack(spectra, axis=0), axis=0)
    freqs = np.fft.rfftfreq(window, 1.0 / sr)
    bands = build_fractional_octave_bands(fraction=smoothing_fraction, start_hz=20.0, stop_hz=20000.0)

    curve = []
    levels = {}
    for band in bands:
        mask = (freqs >= band.low_hz) & (freqs < band.high_hz)
        if not np.any(mask):
            continue
        # Use mean power density inside each fractional-octave band instead of
        # the summed energy. Summation biases wider high-frequency bands upward
        # and makes the compensated +4.5 dB/oct tilt read brighter than the
        # actual monitor analyzer view.
        raw_level_db = 10.0 * np.log10(float(np.mean(avg_power[mask])) + 1e-12)
        compensated_level_db = raw_level_db + compensation_db_per_octave * math.log2(max(band.center_hz, 20.0) / 100.0)
        entry = {
            "center_hz": round(float(band.center_hz), 2),
            "raw_db": round(float(raw_level_db), 2),
            "compensated_db": round(float(compensated_level_db), 2),
        }
        curve.append(entry)
        levels[band.center_hz] = float(compensated_level_db)

    def region(lo: float, hi: float) -> float:
        values = [level for hz, level in levels.items() if lo <= hz < hi]
        if not values:
            return -100.0
        return float(np.mean(values))

    plateau_values = [level for hz, level in levels.items() if 90.0 <= hz < 4500.0]
    plateau_db = float(np.mean(plateau_values)) if plateau_values else -100.0
    regions_db = {
        "infra_20_35": round(region(20.0, 35.0), 2),
        "sub_35_60": round(region(35.0, 60.0), 2),
        "weight_60_120": round(region(60.0, 120.0), 2),
        "plateau_90_4500": round(plateau_db, 2),
        "high_4500_12000": round(region(4500.0, 12000.0), 2),
        "air_12000_20000": round(region(12000.0, 20000.0), 2),
    }

    return {
        "compensation_db_per_octave": compensation_db_per_octave,
        "window_sec": round(window_sec, 2),
        "segments": len(starts),
        "smoothing": f"1/{smoothing_fraction} octave",
        "regions_db": regions_db,
        "weight_60_120_vs_plateau_db": round(regions_db["weight_60_120"] - regions_db["plateau_90_4500"], 2),
        "sub_35_60_vs_plateau_db": round(regions_db["sub_35_60"] - regions_db["plateau_90_4500"], 2),
        "high_4500_12000_vs_plateau_db": round(regions_db["high_4500_12000"] - regions_db["plateau_90_4500"], 2),
        "plateau_spread_db": round(
            float(max(plateau_values) - min(plateau_values)) if plateau_values else 0.0,
            2,
        ),
        "curve": curve,
    }


def _stem_mix_snapshot(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    render_cache: OfflineRenderCache | None = None,
    analysis_preview_sec: float = ANALYZER_RENDER_PREVIEW_SEC,
) -> dict[str, Any]:
    rendered_channels = {
        channel: render_channel_preview_cached(
            channel,
            plan,
            sr,
            preview_sec=analysis_preview_sec,
            render_cache=render_cache,
        )
        for channel, plan in plans.items()
        if not plan.muted
    }
    stem_audio, stem_channels, stem_files = _build_rendered_stem_groups(rendered_channels, plans)
    (
        channel_features,
        stem_features,
        contribution_matrix,
        _channel_stems,
        _channel_priorities,
        _channel_measurements,
    ) = _build_autofoh_measurement_snapshot(rendered_channels, plans, sr)

    stems_report = []
    for stem_name in sorted(stem_audio):
        audio = stem_audio[stem_name]
        mono = mono_sum(audio)
        basic_metrics = metrics_for(mono, sr, instrument="custom")
        features = stem_features.get(stem_name)
        mix_indexes = getattr(features, "mix_indexes", None)
        stems_report.append({
            "stem": stem_name,
            "channel_count": len(stem_channels.get(stem_name, [])),
            "channels": stem_channels.get(stem_name, []),
            "files": stem_files.get(stem_name, []),
            "rms_db": round(float(basic_metrics.get("rms_db", -100.0)), 2),
            "dynamic_range_db": round(float(basic_metrics.get("dynamic_range_db", 0.0)), 2),
            "crest_factor_db": round(float(getattr(features, "crest_factor_db", basic_metrics.get("dynamic_range_db", 0.0))), 2),
            "band_energy": {
                band_name: round(float(level_db), 2)
                for band_name, level_db in basic_metrics.get("band_energy", {}).items()
            },
            "mix_indexes": {
                "sub_index": round(float(getattr(mix_indexes, "sub_index", 0.0)), 2),
                "bass_index": round(float(getattr(mix_indexes, "bass_index", 0.0)), 2),
                "body_index": round(float(getattr(mix_indexes, "body_index", 0.0)), 2),
                "mud_index": round(float(getattr(mix_indexes, "mud_index", 0.0)), 2),
                "presence_index": round(float(getattr(mix_indexes, "presence_index", 0.0)), 2),
                "harshness_index": round(float(getattr(mix_indexes, "harshness_index", 0.0)), 2),
                "sibilance_index": round(float(getattr(mix_indexes, "sibilance_index", 0.0)), 2),
                "air_index": round(float(getattr(mix_indexes, "air_index", 0.0)), 2),
            } if mix_indexes is not None else {},
        })

    band_hierarchy = []
    if contribution_matrix is not None:
        for band_name, row in sorted(contribution_matrix.band_contributions.items()):
            ordered = sorted(row.items(), key=lambda item: item[1], reverse=True)
            dominant_stem, dominant_share = ordered[0] if ordered else ("", 0.0)
            runner_up_stem, runner_up_share = ordered[1] if len(ordered) > 1 else ("", 0.0)
            band_hierarchy.append({
                "band": band_name,
                "dominant_stem": dominant_stem,
                "dominant_share": round(float(dominant_share), 4),
                "runner_up_stem": runner_up_stem,
                "runner_up_share": round(float(runner_up_share), 4),
            })

    master_mix_indexes = {}
    for item in stems_report:
        if item["stem"] == "MASTER":
            master_mix_indexes = dict(item.get("mix_indexes", {}))
            break

    slope_conformity = {
        "reference_tilt_db_per_octave": 4.5,
        "master_mix_indexes": master_mix_indexes,
        "sub_deficit_db": round(max(0.0, -float(master_mix_indexes.get("sub_index", 0.0))), 2),
        "bass_deficit_db": round(max(0.0, -float(master_mix_indexes.get("bass_index", 0.0))), 2),
        "body_deficit_db": round(max(0.0, -float(master_mix_indexes.get("body_index", 0.0))), 2),
    }

    kick_focus = {}
    kick_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "kick"),
        None,
    )
    if kick_entry:
        kick_channel, kick_plan = kick_entry
        kick_audio = rendered_channels.get(kick_channel)
        drums_audio = stem_audio.get("DRUMS")
        master_audio = stem_audio.get("MASTER")
        if kick_audio is not None and drums_audio is not None and master_audio is not None:
            activity = kick_plan.event_activity or _event_activity_ranges(mono_sum(kick_audio), sr, "kick") or {}
            ranges = activity.get("ranges") or []
            kick_focus_audio = _activity_window_audio(kick_audio, ranges, sr)
            drums_focus_audio = _activity_window_audio(drums_audio, ranges, sr)
            master_focus_audio = _activity_window_audio(master_audio, ranges, sr)
            kick_metrics = metrics_for(kick_focus_audio, sr, instrument="kick")
            kick_punch_power = _band_power(kick_focus_audio, sr, 55.0, 95.0)
            kick_click_power = _band_power(kick_focus_audio, sr, 2500.0, 4500.0)
            drums_punch_power = _band_power(drums_focus_audio, sr, 55.0, 95.0)
            drums_click_power = _band_power(drums_focus_audio, sr, 2500.0, 4500.0)
            master_click_power = _band_power(master_focus_audio, sr, 2500.0, 4500.0)
            kick_focus = {
                "channel": kick_channel,
                "file": kick_plan.path.name,
                "analysis_ranges": len(ranges),
                "active_sec": round(float((activity.get("active_samples") or 0) / sr), 3) if sr else 0.0,
                "kick_dynamic_range_db": round(float(kick_metrics.get("dynamic_range_db", 0.0)), 2),
                "kick_punch_db": round(_band_rms_db(kick_focus_audio, sr, 55.0, 95.0), 2),
                "kick_click_db": round(_band_rms_db(kick_focus_audio, sr, 2500.0, 4500.0), 2),
                "kick_box_db": round(_band_rms_db(kick_focus_audio, sr, 220.0, 420.0), 2),
                "kick_click_minus_punch_db": round(
                    _band_rms_db(kick_focus_audio, sr, 2500.0, 4500.0)
                    - _band_rms_db(kick_focus_audio, sr, 55.0, 95.0),
                    2,
                ),
                "kick_click_share_in_drums": round(float(kick_click_power / max(drums_click_power, 1e-12)), 4),
                "kick_punch_share_in_drums": round(float(kick_punch_power / max(drums_punch_power, 1e-12)), 4),
                "kick_click_share_in_master": round(float(kick_click_power / max(master_click_power, 1e-12)), 4),
            }

    tilt_conformity = {
        stem_name: _ltas_tilt_profile(audio, sr)
        for stem_name, audio in stem_audio.items()
        if stem_name in {"MASTER", "DRUMS", "MUSIC", "LEAD", "BASS", "KICK"}
    }

    def _sum_rendered_by_instrument(instruments: set[str]) -> np.ndarray | None:
        selected = [
            rendered_channels[channel]
            for channel, plan in plans.items()
            if channel in rendered_channels and not plan.muted and plan.instrument in instruments
        ]
        if not selected:
            return None
        return sum(selected, np.zeros_like(selected[0]))

    hierarchy_metrics: dict[str, float] = {}
    lead_audio = _sum_rendered_by_instrument({"lead_vocal"})
    bgv_audio = _sum_rendered_by_instrument({"backing_vocal"})
    if lead_audio is not None and bgv_audio is not None:
        hierarchy_metrics["lead_over_bgv_rms_db"] = round(
            metrics_for(mono_sum(lead_audio), sr, instrument="lead_vocal").get("rms_db", -100.0)
            - metrics_for(mono_sum(bgv_audio), sr, instrument="backing_vocal").get("rms_db", -100.0),
            2,
        )
        hierarchy_metrics["lead_over_bgv_presence_db"] = round(
            _band_rms_db(lead_audio, sr, 2200.0, 5200.0) - _band_rms_db(bgv_audio, sr, 2200.0, 5200.0),
            2,
        )

    kick_audio = _sum_rendered_by_instrument({"kick"})
    bass_audio = _sum_rendered_by_instrument({"bass_guitar", "bass", "bass_di", "bass_mic", "synth_bass"})
    if kick_audio is not None and bass_audio is not None:
        hierarchy_metrics["kick_over_bass_55_95_db"] = round(
            _band_rms_db(kick_audio, sr, 55.0, 95.0) - _band_rms_db(bass_audio, sr, 55.0, 95.0),
            2,
        )
        hierarchy_metrics["kick_click_over_bass_2k5_4k5_db"] = round(
            _band_rms_db(kick_audio, sr, 2500.0, 4500.0) - _band_rms_db(bass_audio, sr, 2500.0, 4500.0),
            2,
        )

    cymbal_audio = _sum_rendered_by_instrument({"overhead", "hi_hat", "ride", "oh_l", "oh_r"})
    master_audio = stem_audio.get("MASTER")
    if cymbal_audio is not None and master_audio is not None:
        cymbal_high_power = _band_power(cymbal_audio, sr, 4500.0, 12000.0)
        master_high_power = _band_power(master_audio, sr, 4500.0, 12000.0)
        hierarchy_metrics["cymbal_high_share_in_master"] = round(float(cymbal_high_power / max(master_high_power, 1e-12)), 4)

    return {
        "rendered_channels": rendered_channels,
        "stem_audio": stem_audio,
        "stems": stems_report,
        "band_hierarchy": band_hierarchy,
        "slope_conformity": slope_conformity,
        "tilt_conformity": tilt_conformity,
        "kick_focus": kick_focus,
        "hierarchy_metrics": hierarchy_metrics,
        "analysis_preview_sec": round(float(analysis_preview_sec), 2),
    }


def _reference_distance_from_snapshot(
    snapshot: dict[str, Any],
    reference_context: ReferenceMixContext | None,
    *,
    targets_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets = copy.deepcopy(targets_override) if targets_override else _reference_targets_from_context(reference_context)
    if not targets:
        return {"enabled": False}

    current_tilt = (snapshot.get("tilt_conformity") or {}).get("MASTER", {})
    current_hierarchy = snapshot.get("hierarchy_metrics") or {}
    current_kick = snapshot.get("kick_focus") or {}
    components: dict[str, float] = {}
    current: dict[str, float] = {}
    target: dict[str, float] = {}

    tilt_targets = targets.get("tilt") or {}
    for key in ("weight_60_120_vs_plateau_db", "sub_35_60_vs_plateau_db", "high_4500_12000_vs_plateau_db", "plateau_spread_db"):
        if key not in tilt_targets:
            continue
        current_value = float(current_tilt.get(key, 0.0))
        target_value = float(tilt_targets[key])
        components[key] = abs(current_value - target_value)
        current[key] = round(current_value, 3)
        target[key] = round(target_value, 3)

    hierarchy_targets = targets.get("hierarchy") or {}
    if "lead_over_bgv_rms_db" in hierarchy_targets and "lead_over_bgv_rms_db" in current_hierarchy:
        current_value = float(current_hierarchy["lead_over_bgv_rms_db"])
        target_value = float(hierarchy_targets["lead_over_bgv_rms_db"])
        components["lead_over_bgv_rms_db"] = abs(current_value - target_value)
        current["lead_over_bgv_rms_db"] = round(current_value, 3)
        target["lead_over_bgv_rms_db"] = round(target_value, 3)
    if "kick_over_bass_55_95_db" in hierarchy_targets and "kick_over_bass_55_95_db" in current_hierarchy:
        current_value = float(current_hierarchy["kick_over_bass_55_95_db"])
        target_value = float(hierarchy_targets["kick_over_bass_55_95_db"])
        components["kick_over_bass_55_95_db"] = abs(current_value - target_value)
        current["kick_over_bass_55_95_db"] = round(current_value, 3)
        target["kick_over_bass_55_95_db"] = round(target_value, 3)

    kick_targets = targets.get("kick") or {}
    if "dynamic_range_max_db" in kick_targets and "kick_dynamic_range_db" in current_kick:
        current_value = float(current_kick["kick_dynamic_range_db"])
        target_value = float(kick_targets["dynamic_range_max_db"])
        components["kick_dynamic_range_db"] = abs(current_value - target_value)
        current["kick_dynamic_range_db"] = round(current_value, 3)
        target["kick_dynamic_range_db"] = round(target_value, 3)
    if "click_share_in_master" in kick_targets and "kick_click_share_in_master" in current_kick:
        current_value = float(current_kick["kick_click_share_in_master"])
        target_value = float(kick_targets["click_share_in_master"])
        components["kick_click_share_in_master"] = abs(current_value - target_value) * 100.0
        current["kick_click_share_in_master"] = round(current_value, 4)
        target["kick_click_share_in_master"] = round(target_value, 4)
    if "click_share_in_drums" in kick_targets and "kick_click_share_in_drums" in current_kick:
        current_value = float(current_kick["kick_click_share_in_drums"])
        target_value = float(kick_targets["click_share_in_drums"])
        components["kick_click_share_in_drums"] = abs(current_value - target_value) * 100.0
        current["kick_click_share_in_drums"] = round(current_value, 4)
        target["kick_click_share_in_drums"] = round(target_value, 4)

    overall = float(np.mean(list(components.values()))) if components else 0.0
    return {
        "enabled": True,
        "overall_distance": round(overall, 4),
        "components": {key: round(value, 4) for key, value in components.items()},
        "current": current,
        "target": target,
    }


def apply_frequency_window_balance(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    render_cache: OfflineRenderCache | None = None,
) -> dict[str, Any]:
    rendered_before = {
        channel: render_channel_preview_cached(
            channel,
            plan,
            sr,
            preview_sec=FREQUENCY_WINDOW_PREVIEW_SEC,
            render_cache=render_cache,
        )
        for channel, plan in plans.items()
        if not plan.muted
    }
    snapshot_before = _frequency_window_snapshot(rendered_before, plans, sr)
    by_id = snapshot_before.get("by_id") or {}
    config_by_id = {item["id"]: item for item in FREQUENCY_WINDOW_DEFINITIONS}
    actions: list[dict[str, Any]] = []

    def _candidate_channels(
        window_id: str,
        *,
        instruments: set[str],
        min_share: float = 0.06,
    ) -> list[dict[str, Any]]:
        window = by_id.get(window_id) or {}
        return [
            item for item in (window.get("channels") or [])
            if item.get("instrument") in instruments and float(item.get("share", 0.0)) >= min_share
        ]

    def _apply_window_cut(
        window_id: str,
        *,
        channels: list[dict[str, Any]],
        base_cut_db: float,
        reason: str,
    ) -> None:
        if not channels:
            return
        config = config_by_id.get(window_id) or {}
        center_hz = float(config.get("center_hz", 1000.0))
        q = float(config.get("q", 1.2))
        strongest_share = max(float(channels[0].get("share", 0.0)), 1e-6)
        for item in channels[:3]:
            channel = int(item["channel"])
            plan = plans.get(channel)
            if plan is None or plan.muted:
                continue
            scale = math.sqrt(max(float(item.get("share", 0.0)), 1e-6) / strongest_share)
            cut_db = float(np.clip(base_cut_db * scale, 0.25, base_cut_db))
            plan.eq_bands.append((center_hz, -cut_db, q))
            actions.append({
                "type": "frequency_window_eq_cut",
                "window": window_id,
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "frequency_hz": round(center_hz, 1),
                "gain_db": round(-cut_db, 2),
                "q": round(q, 2),
                "share_before": round(float(item.get("share", 0.0)), 4),
                "reason": reason,
            })

    warmth = by_id.get("warmth_mud") or {}
    warmth_shares = warmth.get("family_shares") or {}
    warmth_music_share = float(warmth_shares.get("music", 0.0) + warmth_shares.get("backing_vocal", 0.0))
    warmth_lead_share = float(warmth_shares.get("lead_vocal", 0.0))
    if warmth_music_share > 0.58 and warmth_lead_share < 0.24:
        warmth_cut_db = float(np.clip(0.35 + (warmth_music_share - max(warmth_lead_share, 0.16)) * 0.85, 0.35, 0.95))
        _apply_window_cut(
            "warmth_mud",
            channels=_candidate_channels("warmth_mud", instruments=WINDOW_SPACE_COMPETITOR_INSTRUMENTS),
            base_cut_db=warmth_cut_db,
            reason="120-500 Hz window is crowded by accompaniment body; broad low-mid cleanup restores depth without thinning the full mix.",
        )

    vocal_conflict = by_id.get("vocal_conflict") or {}
    vocal_shares = vocal_conflict.get("family_shares") or {}
    vocal_competition_share = float(vocal_shares.get("music", 0.0) + vocal_shares.get("backing_vocal", 0.0))
    vocal_lead_share = float(vocal_shares.get("lead_vocal", 0.0))
    vocal_window_advantage = vocal_competition_share - vocal_lead_share
    if (
        vocal_competition_share > 0.54
        and (
            vocal_lead_share < 0.26
            or str(vocal_conflict.get("dominant_family") or "") != "lead_vocal"
            or vocal_window_advantage > 0.045
        )
    ):
        vocal_cut_db = float(np.clip(0.4 + max(vocal_window_advantage, 0.0) * 1.55, 0.4, 1.2))
        _apply_window_cut(
            "vocal_conflict",
            channels=_candidate_channels("vocal_conflict", instruments=WINDOW_SPACE_COMPETITOR_INSTRUMENTS),
            base_cut_db=vocal_cut_db,
            reason="700-1500 Hz window hides the lead behind accompaniment; this carve clears the speaking range instead of just raising the vocal.",
        )

    presence = by_id.get("presence_harshness") or {}
    presence_shares = presence.get("family_shares") or {}
    presence_competition_share = float(presence_shares.get("music", 0.0) + presence_shares.get("backing_vocal", 0.0))
    presence_lead_share = float(presence_shares.get("lead_vocal", 0.0))
    if presence_competition_share > 0.52 and presence_lead_share < 0.24:
        presence_cut_db = float(np.clip(0.35 + (presence_competition_share - max(presence_lead_share, 0.18)) * 0.95, 0.35, 1.0))
        _apply_window_cut(
            "presence_harshness",
            channels=_candidate_channels("presence_harshness", instruments=WINDOW_SPACE_COMPETITOR_INSTRUMENTS, min_share=0.05),
            base_cut_db=presence_cut_db,
            reason="1.5-6 kHz window is overfilled by accompaniment presence; broad presence carving helps lyric intelligibility without global brightening.",
        )

    air = by_id.get("air_sibilance") or {}
    air_shares = air.get("family_shares") or {}
    cymbal_share = float(air_shares.get("cymbals", 0.0))
    if cymbal_share > 0.30:
        air_cut_db = float(np.clip(0.35 + (cymbal_share - 0.30) * 2.0, 0.35, 1.15))
        _apply_window_cut(
            "air_sibilance",
            channels=_candidate_channels("air_sibilance", instruments=CYMBAL_INSTRUMENTS, min_share=0.05),
            base_cut_db=air_cut_db,
            reason="6-16 kHz window is dominated by cymbal wash; a narrow upper-air trim preserves sparkle while reducing constant hiss.",
        )

    if not actions:
        return {
            "enabled": True,
            "applied": False,
            "actions": [],
            "notes": [
                "Frequency-window balancing analyses broad musical windows rather than chasing narrow resonances.",
                "Use these windows as a diagnostic magnifier: low end, warmth, core mids, vocal conflict, presence, and air.",
            ],
            "before": snapshot_before["windows"],
            "after": snapshot_before["windows"],
        }

    rendered_after = {
        channel: render_channel_preview_cached(
            channel,
            plan,
            sr,
            preview_sec=FREQUENCY_WINDOW_PREVIEW_SEC,
            render_cache=render_cache,
        )
        for channel, plan in plans.items()
        if not plan.muted
    }
    snapshot_after = _frequency_window_snapshot(rendered_after, plans, sr)
    return {
        "enabled": True,
        "applied": True,
        "actions": actions,
        "notes": [
            "Frequency-window balancing analyses broad musical windows rather than chasing narrow resonances.",
            "Each corrective move is a bounded wide-band carve applied only to the competing sources inside the overloaded window.",
            "Low end stays report-only here because kick/bass hierarchy already has its own dedicated measurement pass.",
        ],
        "before": snapshot_before["windows"],
        "after": snapshot_after["windows"],
    }


def apply_stem_mix_verification(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    genre: str | None = None,
    reference_context: ReferenceMixContext | None = None,
    render_cache: OfflineRenderCache | None = None,
    analysis_preview_sec: float = ANALYZER_RENDER_PREVIEW_SEC,
) -> dict[str, Any]:
    snapshot_before = _stem_mix_snapshot(
        plans,
        target_len,
        sr,
        render_cache=render_cache,
        analysis_preview_sec=analysis_preview_sec,
    )
    genre_token = _normalize_genre_token(genre)
    reference_targets = _effective_balance_targets(reference_context, genre=genre)
    reference_distance_before = _reference_distance_from_snapshot(
        snapshot_before,
        reference_context,
        targets_override=reference_targets,
    )
    kick_focus_before = snapshot_before.get("kick_focus") or {}
    kick_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "kick"),
        None,
    )
    if not kick_entry:
        return {
            "enabled": False,
            "reason": "kick_missing",
            "before": {
                "stems": snapshot_before["stems"],
                "band_hierarchy": snapshot_before["band_hierarchy"],
                "slope_conformity": snapshot_before["slope_conformity"],
                "tilt_conformity": snapshot_before["tilt_conformity"],
                "kick_focus": kick_focus_before,
            },
        }

    _, kick_plan = kick_entry
    desired_click_share_in_drums = 0.14 if genre_token == "rock" else 0.075
    desired_click_share_in_master = 0.045 if genre_token == "rock" else 0.022
    desired_dynamic_range_max_db = 16.5 if genre_token == "rock" else 18.0
    desired_click_minus_punch_db = -17.0 if genre_token == "rock" else -20.0
    desired_box_minus_click_db = -1.5 if genre_token == "rock" else 1.5
    if reference_targets:
        kick_targets = reference_targets.get("kick") or {}
        desired_click_share_in_drums = float(kick_targets.get("click_share_in_drums", desired_click_share_in_drums))
        desired_click_share_in_master = float(kick_targets.get("click_share_in_master", desired_click_share_in_master))
        desired_dynamic_range_max_db = float(kick_targets.get("dynamic_range_max_db", desired_dynamic_range_max_db))
        desired_click_minus_punch_db = float(kick_targets.get("click_minus_punch_db", desired_click_minus_punch_db))
        desired_box_minus_click_db = float(kick_targets.get("box_minus_click_db", desired_box_minus_click_db))
    slope_before = snapshot_before.get("slope_conformity") or {}
    master_mix_indexes_before = slope_before.get("master_mix_indexes") or {}
    master_tilt_before = (snapshot_before.get("tilt_conformity") or {}).get("MASTER", {})
    master_tilt_regions_before = master_tilt_before.get("regions_db") or {}
    hierarchy_before = snapshot_before.get("hierarchy_metrics") or {}

    click_share_drums = float(kick_focus_before.get("kick_click_share_in_drums", 0.0))
    click_share_master = float(kick_focus_before.get("kick_click_share_in_master", 0.0))
    kick_dynamic_range_db = float(kick_focus_before.get("kick_dynamic_range_db", 0.0))
    click_minus_punch_db = float(kick_focus_before.get("kick_click_minus_punch_db", -100.0))
    box_minus_click_db = float(kick_focus_before.get("kick_box_db", -100.0) - kick_focus_before.get("kick_click_db", -100.0))
    weight_vs_plateau_db = float(master_tilt_before.get("weight_60_120_vs_plateau_db", 0.0))
    sub_vs_plateau_db = float(master_tilt_before.get("sub_35_60_vs_plateau_db", 0.0))
    high_vs_plateau_db = float(master_tilt_before.get("high_4500_12000_vs_plateau_db", 0.0))
    plateau_spread_db = float(master_tilt_before.get("plateau_spread_db", 0.0))
    sub_index_before = float(master_mix_indexes_before.get("sub_index", 0.0))
    bass_index_before = float(master_mix_indexes_before.get("bass_index", 0.0))
    body_index_before = float(master_mix_indexes_before.get("body_index", 0.0))
    low_end_deficit = max(
        0.0,
        (-sub_index_before) - (0.8 if genre_token == "rock" else 1.1),
        (-bass_index_before) - (0.6 if genre_token == "rock" else 0.9),
        (-body_index_before) - (0.35 if genre_token == "rock" else 0.6),
    )
    target_weight_vs_plateau_db = 2.0 if genre_token == "rock" else 0.35
    target_sub_vs_plateau_db = 0.5 if genre_token == "rock" else -0.25
    target_high_vs_plateau_db = -1.5
    target_plateau_spread_db = 6.0
    if reference_targets:
        tilt_targets = reference_targets.get("tilt") or {}
        target_weight_vs_plateau_db = float(tilt_targets.get("weight_60_120_vs_plateau_db", target_weight_vs_plateau_db))
        target_sub_vs_plateau_db = float(tilt_targets.get("sub_35_60_vs_plateau_db", target_sub_vs_plateau_db))
        target_high_vs_plateau_db = float(tilt_targets.get("high_4500_12000_vs_plateau_db", target_high_vs_plateau_db))
        target_plateau_spread_db = float(tilt_targets.get("plateau_spread_db", target_plateau_spread_db))
    tilt_weight_shortage = max(0.0, target_weight_vs_plateau_db - weight_vs_plateau_db)
    tilt_sub_shortage = max(0.0, target_sub_vs_plateau_db - sub_vs_plateau_db)
    tilt_brightness_excess = max(0.0, high_vs_plateau_db - target_high_vs_plateau_db)
    plateau_unevenness = max(0.0, plateau_spread_db - target_plateau_spread_db)

    click_shortage = max(
        0.0,
        desired_click_share_in_drums - click_share_drums,
        (desired_click_share_in_master - click_share_master) * 1.8,
    )
    dynamic_shortage = max(0.0, kick_dynamic_range_db - desired_dynamic_range_max_db)
    click_tone_shortage = max(0.0, desired_click_minus_punch_db - click_minus_punch_db)
    box_excess = max(0.0, box_minus_click_db - desired_box_minus_click_db)

    actions: list[dict[str, Any]] = []
    hierarchy_targets = reference_targets.get("hierarchy") or {}
    lead_gap_target = float(hierarchy_targets.get("lead_over_bgv_rms_db", 4.2))
    lead_gap_before = float(hierarchy_before.get("lead_over_bgv_rms_db", 0.0))
    lead_gap_shortage = max(0.0, lead_gap_target - lead_gap_before)
    lead_gap_excess = max(0.0, lead_gap_before - (lead_gap_target + 0.9))
    if lead_gap_shortage > 0.7:
        lead_channels = [
            channel for channel, plan in plans.items()
            if not plan.muted and plan.instrument == "lead_vocal"
        ]
        bgv_channels = [
            channel for channel, plan in plans.items()
            if not plan.muted and plan.instrument == "backing_vocal"
        ]
        if lead_channels:
            lift_db = float(np.clip(0.35 + lead_gap_shortage * 0.22, 0.35, 1.0))
            for channel in lead_channels:
                plan = plans[channel]
                before = float(plan.fader_db)
                plan.fader_db = float(np.clip(plan.fader_db + lift_db, -30.0, 0.0))
                if plan.fader_db != before:
                    actions.append({
                        "type": "reference_lead_foreground",
                        "channel": channel,
                        "before": {"fader_db": round(before, 2)},
                        "after": {"fader_db": round(float(plan.fader_db), 2)},
                        "reason": "Reference hierarchy expects lead vocals to sit further ahead of the backing stack.",
                    })
        if bgv_channels:
            bgv_cut_db = float(np.clip(0.45 + lead_gap_shortage * 0.28, 0.45, 1.4))
            for channel in bgv_channels:
                plan = plans[channel]
                before = float(plan.fader_db)
                plan.fader_db = float(np.clip(plan.fader_db - bgv_cut_db, -30.0, 0.0))
                if plan.fader_db != before:
                    actions.append({
                        "type": "reference_bgv_depth",
                        "channel": channel,
                        "before": {"fader_db": round(before, 2)},
                        "after": {"fader_db": round(float(plan.fader_db), 2)},
                        "reason": "Reference hierarchy keeps backing vocals behind the lead image instead of flattening the vocal stack.",
                    })
    elif lead_gap_excess > 0.7:
        lead_channels = [
            channel for channel, plan in plans.items()
            if not plan.muted and plan.instrument == "lead_vocal"
        ]
        bgv_channels = [
            channel for channel, plan in plans.items()
            if not plan.muted and plan.instrument == "backing_vocal"
        ]
        if bgv_channels:
            bgv_lift_db = float(np.clip(0.55 + lead_gap_excess * 0.26, 0.55, 1.8))
            for channel in bgv_channels:
                plan = plans[channel]
                before = float(plan.fader_db)
                plan.fader_db = float(np.clip(plan.fader_db + bgv_lift_db, -30.0, 0.0))
                if plan.fader_db != before:
                    actions.append({
                        "type": "reference_bgv_pull_forward",
                        "channel": channel,
                        "before": {"fader_db": round(before, 2)},
                        "after": {"fader_db": round(float(plan.fader_db), 2)},
                        "reason": "Reference hierarchy keeps the lead closer to the backing stack than the current mix.",
                    })
        if lead_channels:
            lead_trim_db = float(np.clip(0.35 + lead_gap_excess * 0.14, 0.35, 1.0))
            for channel in lead_channels:
                plan = plans[channel]
                before = float(plan.fader_db)
                plan.fader_db = float(np.clip(plan.fader_db - lead_trim_db, -30.0, 0.0))
                if plan.fader_db != before:
                    actions.append({
                        "type": "reference_lead_relax",
                        "channel": channel,
                        "before": {"fader_db": round(before, 2)},
                        "after": {"fader_db": round(float(plan.fader_db), 2)},
                        "reason": "Reference hierarchy does not want the lead so detached from the backing image.",
                    })

    if click_shortage > 0.0 or dynamic_shortage > 0.0 or click_tone_shortage > 0.0 or box_excess > 0.0:
        before_comp = {
            "threshold_db": round(float(kick_plan.comp_threshold_db), 2),
            "ratio": round(float(kick_plan.comp_ratio), 2),
            "attack_ms": round(float(kick_plan.comp_attack_ms), 2),
            "release_ms": round(float(kick_plan.comp_release_ms), 2),
        }
        kick_plan.comp_threshold_db = float(min(kick_plan.comp_threshold_db, -23.5 if genre_token == "rock" else -22.0))
        kick_plan.comp_ratio = float(max(kick_plan.comp_ratio, 5.4 if genre_token == "rock" else 4.8))
        kick_plan.comp_attack_ms = float(np.clip(max(kick_plan.comp_attack_ms, 10.0), 10.0, 12.0))
        kick_plan.comp_release_ms = float(np.clip(kick_plan.comp_release_ms, 80.0, 95.0))
        actions.append({
            "type": "kick_compressor_reseat",
            "before": before_comp,
            "after": {
                "threshold_db": round(float(kick_plan.comp_threshold_db), 2),
                "ratio": round(float(kick_plan.comp_ratio), 2),
                "attack_ms": round(float(kick_plan.comp_attack_ms), 2),
                "release_ms": round(float(kick_plan.comp_release_ms), 2),
            },
            "reason": "Kick stem stays too spiky relative to the drum/master stems and needs firmer body control without choking the click.",
        })

    if click_shortage > 0.0 or click_tone_shortage > 0.0:
        click_gain_db = float(
            np.clip(
                0.8 + click_shortage * 12.0 + click_tone_shortage * 0.08,
                0.8,
                2.1 if genre_token == "rock" else 1.45,
            )
        )
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=3500.0,
            gain_db=click_gain_db,
            q=1.6,
        )
        if action is not None:
            actions.append({
                "type": "kick_click_boost",
                "target": {"freq_hz": 3500.0, "gain_db": round(click_gain_db, 2), "q": 1.6},
                "result": action,
                "reason": "Kick click share inside the drum and master stems is too low.",
            })

    if genre_token == "rock" and click_shortage > 0.015:
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=2200.0,
            gain_db=0.6,
            q=1.4,
        )
        if action is not None:
            actions.append({
                "type": "kick_upper_attack_support",
                "target": {"freq_hz": 2200.0, "gain_db": 0.6, "q": 1.4},
                "result": action,
                "reason": "Kick needs more upper attack definition so the beater survives inside the drum stem.",
            })

    if box_excess > 0.0:
        box_cut_db = float(np.clip(-0.6 - box_excess * 0.15, -1.4, -0.6))
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=320.0,
            gain_db=box_cut_db,
            q=1.25,
        )
        if action is not None:
            actions.append({
                "type": "kick_box_cleanup",
                "target": {"freq_hz": 320.0, "gain_db": round(box_cut_db, 2), "q": 1.25},
                "result": action,
                "reason": "Kick box energy is overtaking the click band and keeps the drum from sitting forward in the mix.",
            })

    if low_end_deficit > 0.0 or tilt_weight_shortage > 0.0 or tilt_sub_shortage > 0.0:
        low_end_support = max(low_end_deficit * 0.12, tilt_weight_shortage * 0.35, tilt_sub_shortage * 0.25)
        kick_low_gain_db = float(np.clip(0.75 + low_end_support, 0.75, 2.4 if genre_token == "rock" else 1.45))
        action = _merge_reference_eq_adjustment(
            kick_plan,
            freq=62.0,
            gain_db=kick_low_gain_db,
            q=0.95,
        )
        if action is not None:
            actions.append({
                "type": "master_slope_low_end_support_kick",
                "target": {"freq_hz": 62.0, "gain_db": round(kick_low_gain_db, 2), "q": 0.95},
                "result": action,
                "reason": "AutoFOH slope-conformity check found a low-end deficit versus the -4.5 dB/oct target line.",
            })

        bass_entry = next(
            ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "bass_guitar"),
            None,
        )
        if bass_entry is not None:
            _, bass_plan = bass_entry
            bass_low_gain_db = float(np.clip(0.55 + max(low_end_deficit * 0.08, tilt_weight_shortage * 0.25), 0.55, 1.8))
            action = _merge_reference_eq_adjustment(
                bass_plan,
                freq=78.0,
                gain_db=bass_low_gain_db,
                q=1.0,
            )
            if action is not None:
                actions.append({
                    "type": "master_slope_low_end_support_bass",
                    "target": {"freq_hz": 78.0, "gain_db": round(bass_low_gain_db, 2), "q": 1.0},
                    "result": action,
                    "reason": "Bass stem gets bounded support so the final mix better conforms to the -4.5 dB/oct slope target without giving away kick priority.",
                })
            if max(low_end_deficit, tilt_weight_shortage) > 4.0 and bass_plan.fader_db < -0.6:
                before_fader = float(bass_plan.fader_db)
                bass_plan.fader_db = float(np.clip(bass_plan.fader_db + 0.55, -30.0, 0.0))
                if bass_plan.fader_db != before_fader:
                    actions.append({
                        "type": "master_slope_bass_fader_recovery",
                        "before": {"fader_db": round(before_fader, 2)},
                        "after": {"fader_db": round(float(bass_plan.fader_db), 2)},
                        "reason": "Kick remains the low-end leader, but the bass fader recovers a little because the master slope still lacks weight.",
                    })

        if genre_token == "rock" and max(low_end_deficit, tilt_weight_shortage, tilt_sub_shortage) > 4.0 and kick_plan.trim_db < 3.0:
            before_trim = float(kick_plan.trim_db)
            kick_plan.trim_db = float(np.clip(kick_plan.trim_db + 0.6, -18.0, 12.0))
            if kick_plan.trim_db != before_trim:
                actions.append({
                    "type": "master_slope_kick_trim_support",
                    "before": {"trim_db": round(before_trim, 2)},
                    "after": {"trim_db": round(float(kick_plan.trim_db), 2)},
                    "reason": "Kick gets a small pre-compression lift because the master still sits below the target low-end slope.",
                })

    if tilt_brightness_excess > 0.0 or plateau_unevenness > 0.0:
        cymbal_cut_db = float(np.clip(0.45 + tilt_brightness_excess * 0.18 + plateau_unevenness * 0.05, 0.45, 1.2))
        for channel, plan in plans.items():
            if plan.muted or plan.instrument not in {"overhead", "hi_hat", "ride"}:
                continue
            action = _merge_reference_eq_adjustment(
                plan,
                freq=8500.0,
                gain_db=-cymbal_cut_db,
                q=1.0,
            )
            if action is None:
                continue
            actions.append({
                "type": "tilt_high_band_trim",
                "channel": channel,
                "target": {"freq_hz": 8500.0, "gain_db": round(-cymbal_cut_db, 2), "q": 1.0},
                "result": action,
                "reason": "The compensated LTAS stays too bright above 4.5 kHz, so cymbal/top-end stems are trimmed to restore the intended post-5 kHz roll-off.",
            })

    if not actions:
        return {
            "enabled": True,
            "genre": genre_token,
            "reference_targets": reference_targets,
            "reference_distance": {
                "before": reference_distance_before,
                "after": reference_distance_before,
            },
            "applied": False,
            "notes": [
                "Stem mix verification assembled the mix by stems and checked spectral balance, dynamics, and band hierarchy.",
                "Kick stem already has enough click share and controlled dynamics inside the drum/master stems.",
            ],
            "before": {
                "stems": snapshot_before["stems"],
                "band_hierarchy": snapshot_before["band_hierarchy"],
                "slope_conformity": snapshot_before["slope_conformity"],
                "tilt_conformity": snapshot_before["tilt_conformity"],
                "kick_focus": kick_focus_before,
                "hierarchy_metrics": snapshot_before["hierarchy_metrics"],
            },
            "after": {
                "stems": snapshot_before["stems"],
                "band_hierarchy": snapshot_before["band_hierarchy"],
                "slope_conformity": snapshot_before["slope_conformity"],
                "tilt_conformity": snapshot_before["tilt_conformity"],
                "kick_focus": kick_focus_before,
                "hierarchy_metrics": snapshot_before["hierarchy_metrics"],
            },
            "actions": [],
        }

    snapshot_after = _stem_mix_snapshot(
        plans,
        target_len,
        sr,
        render_cache=render_cache,
        analysis_preview_sec=analysis_preview_sec,
    )
    reference_distance_after = _reference_distance_from_snapshot(
        snapshot_after,
        reference_context,
        targets_override=reference_targets,
    )
    return {
        "enabled": True,
        "genre": genre_token,
        "reference_targets": reference_targets,
        "reference_distance": {
            "before": reference_distance_before,
            "after": reference_distance_after,
        },
        "applied": True,
        "notes": [
            "Stem mix verification assembles rendered channels into stems before the final print.",
            "The pass checks stem AChH, dynamics, and band hierarchy, then reseats the kick only if its click/dynamics are weak inside the drum or master stems.",
            "When a reference is present, the same verification also measures distance to the reference hierarchy and tilt targets.",
        ],
        "before": {
            "stems": snapshot_before["stems"],
            "band_hierarchy": snapshot_before["band_hierarchy"],
            "slope_conformity": snapshot_before["slope_conformity"],
            "tilt_conformity": snapshot_before["tilt_conformity"],
            "kick_focus": kick_focus_before,
            "hierarchy_metrics": snapshot_before["hierarchy_metrics"],
        },
        "after": {
            "stems": snapshot_after["stems"],
            "band_hierarchy": snapshot_after["band_hierarchy"],
            "slope_conformity": snapshot_after["slope_conformity"],
            "tilt_conformity": snapshot_after["tilt_conformity"],
            "kick_focus": snapshot_after["kick_focus"],
            "hierarchy_metrics": snapshot_after["hierarchy_metrics"],
        },
        "actions": actions,
    }


def apply_kick_bass_hierarchy(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    *,
    desired_kick_advantage_db: float = 1.0,
    render_cache: OfflineRenderCache | None = None,
    analysis_preview_sec: float = ANALYZER_RENDER_PREVIEW_SEC,
) -> dict[str, Any]:
    kick_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "kick"),
        None,
    )
    bass_entry = next(
        ((channel, plan) for channel, plan in plans.items() if not plan.muted and plan.instrument == "bass_guitar"),
        None,
    )
    if not kick_entry or not bass_entry:
        return {"enabled": False, "reason": "kick_or_bass_missing"}

    kick_channel, kick_plan = kick_entry
    bass_channel, bass_plan = bass_entry
    kick_audio = render_channel_preview_cached(
        kick_channel,
        kick_plan,
        sr,
        preview_sec=analysis_preview_sec,
        render_cache=render_cache,
    )
    bass_audio = render_channel_preview_cached(
        bass_channel,
        bass_plan,
        sr,
        preview_sec=analysis_preview_sec,
        render_cache=render_cache,
    )

    kick_block, kick_meta = _analysis_signal_for_metrics(mono_sum(kick_audio), sr, "kick")
    bass_block, bass_meta = _analysis_signal_for_metrics(mono_sum(bass_audio), sr, "bass_guitar")
    kick_anchor_db = _band_rms_db(kick_block, sr, 55.0, 95.0)
    bass_anchor_db = _band_rms_db(bass_block, sr, 55.0, 95.0)
    kick_click_db = _band_rms_db(kick_block, sr, 2500.0, 4500.0)
    bass_low_mid_db = _band_rms_db(bass_block, sr, 90.0, 160.0)
    measured_advantage = kick_anchor_db - bass_anchor_db
    shortage_db = desired_kick_advantage_db - measured_advantage

    if shortage_db <= 0.25:
        return {
            "enabled": False,
            "reason": "hierarchy_already_satisfied",
            "kick_channel": kick_channel,
            "bass_channel": bass_channel,
            "kick_anchor_db": round(kick_anchor_db, 2),
            "bass_anchor_db": round(bass_anchor_db, 2),
            "measured_advantage_db": round(measured_advantage, 2),
        "desired_kick_advantage_db": round(desired_kick_advantage_db, 2),
        "analysis_preview_sec": round(float(analysis_preview_sec), 2),
    }

    kick_fader_before = float(kick_plan.fader_db)
    bass_fader_before = float(bass_plan.fader_db)
    kick_trim_before = float(kick_plan.trim_db)
    kick_boost_db = float(np.clip(0.8 + shortage_db * 0.45, 0.75, 1.8))
    bass_cut_ceiling_db = 2.8 if shortage_db > 4.0 else 1.4
    bass_cut_db = float(np.clip(0.45 + shortage_db * 0.4, 0.35, bass_cut_ceiling_db))
    kick_plan.fader_db = float(np.clip(kick_plan.fader_db + kick_boost_db, -30.0, 0.0))
    bass_plan.fader_db = float(np.clip(bass_plan.fader_db - bass_cut_db, -30.0, 0.0))
    actual_kick_fader_boost_db = float(kick_plan.fader_db - kick_fader_before)

    kick_trim_boost_db = 0.0
    if actual_kick_fader_boost_db < kick_boost_db - 0.25:
        trim_ceiling_db = min(6.0, float(kick_plan.trim_analysis.get("max_boost_db", 6.0) or 6.0))
        available_trim_boost_db = max(0.0, trim_ceiling_db - float(kick_plan.trim_db))
        requested_trim_boost_db = float(np.clip(0.7 + shortage_db * 0.18, 0.5, 3.2))
        kick_trim_boost_db = min(available_trim_boost_db, requested_trim_boost_db)
        if kick_trim_boost_db > 0.05:
            kick_plan.trim_db = float(np.clip(kick_plan.trim_db + kick_trim_boost_db, -18.0, trim_ceiling_db))

    kick_eq_changes: list[tuple[float, float, float]] = []
    bass_eq_changes: list[tuple[float, float, float]] = []

    if kick_anchor_db < bass_anchor_db + desired_kick_advantage_db:
        kick_eq_changes.append((68.0, float(np.clip(0.8 + shortage_db * 0.35, 0.8, 2.2)), 0.95))
        kick_eq_changes.append((3200.0, float(np.clip(0.6 + shortage_db * 0.08, 0.6, 1.3)), 1.6))
    bass_overlap_db = max(bass_anchor_db, bass_low_mid_db)
    if shortage_db > 0.5 or bass_overlap_db > kick_anchor_db - 1.0:
        bass_eq_changes.append((82.0, float(np.clip(-0.7 - shortage_db * 0.2, -1.3, -0.7)), 1.0))
        bass_eq_changes.append((125.0, -0.6, 1.1))

    kick_plan.eq_bands.extend(kick_eq_changes)
    bass_plan.eq_bands.extend(bass_eq_changes)

    return {
        "enabled": True,
        "kick_channel": kick_channel,
        "bass_channel": bass_channel,
        "kick_file": kick_plan.path.name,
        "bass_file": bass_plan.path.name,
        "kick_analysis_mode": kick_meta.get("analysis_mode"),
        "bass_analysis_mode": bass_meta.get("analysis_mode"),
        "kick_anchor_db": round(kick_anchor_db, 2),
        "bass_anchor_db": round(bass_anchor_db, 2),
        "kick_click_db": round(kick_click_db, 2),
        "bass_low_mid_db": round(bass_low_mid_db, 2),
        "measured_advantage_db": round(measured_advantage, 2),
        "desired_kick_advantage_db": round(desired_kick_advantage_db, 2),
        "analysis_preview_sec": round(float(analysis_preview_sec), 2),
        "shortage_db": round(shortage_db, 2),
        "kick_fader_before_db": round(kick_fader_before, 2),
        "kick_fader_after_db": round(float(kick_plan.fader_db), 2),
        "kick_trim_before_db": round(kick_trim_before, 2),
        "kick_trim_after_db": round(float(kick_plan.trim_db), 2),
        "kick_trim_boost_db": round(float(kick_trim_boost_db), 2),
        "bass_fader_before_db": round(bass_fader_before, 2),
        "bass_fader_after_db": round(float(bass_plan.fader_db), 2),
        "kick_eq_added": [
            {"freq_hz": round(freq, 2), "gain_db": round(gain, 2), "q": round(q, 2)}
            for freq, gain, q in kick_eq_changes
        ],
        "bass_eq_added": [
            {"freq_hz": round(freq, 2), "gain_db": round(gain, 2), "q": round(q, 2)}
            for freq, gain, q in bass_eq_changes
        ],
        "notes": [
            "Kick is protected as the low-end anchor and must stay ahead of bass in the punch band.",
            "Bass is trimmed only in the overlapping 55-125 Hz area so the groove stays intact while the kick leads.",
        ],
    }


def desired_kick_advantage_from_reference(
    reference_context: ReferenceMixContext | None,
    *,
    genre: str | None = None,
) -> float:
    genre_token = _normalize_genre_token(genre)
    default = 1.8 if genre_token == "rock" else 0.85
    targets = _effective_balance_targets(reference_context, genre=genre)
    hierarchy = targets.get("hierarchy") or {}
    if "kick_over_bass_55_95_db" not in hierarchy:
        return float(default)
    target = float(hierarchy["kick_over_bass_55_95_db"])
    return float(np.clip(default * 0.55 + target * 0.45, 0.5, 2.8))


def apply_reference_vocal_fx_focus(
    plans: dict[int, ChannelPlan],
    target_len: int,
    sr: int,
    reference_context: ReferenceMixContext | None,
    genre: str | None = None,
    *,
    render_cache: OfflineRenderCache | None = None,
    analysis_preview_sec: float = ANALYZER_RENDER_PREVIEW_SEC,
) -> dict[str, Any]:
    targets = _effective_balance_targets(reference_context, genre=genre)
    if not targets:
        return {"enabled": False, "reason": "no_reference_targets"}

    lead_entries = [
        (channel, plan)
        for channel, plan in plans.items()
        if not plan.muted and plan.instrument == "lead_vocal"
    ]
    bgv_entries = [
        (channel, plan)
        for channel, plan in plans.items()
        if not plan.muted and plan.instrument == "backing_vocal"
    ]
    if not lead_entries or not bgv_entries:
        return {"enabled": False, "reason": "lead_or_bgv_missing"}

    def _sum_render(entries: list[tuple[int, ChannelPlan]]) -> np.ndarray:
        rendered = [
            render_channel_preview_cached(
                channel,
                plan,
                sr,
                preview_sec=analysis_preview_sec,
                render_cache=render_cache,
            )
            for channel, plan in entries
        ]
        return sum(rendered, np.zeros_like(rendered[0]))

    def _measure_gap() -> dict[str, float]:
        lead_audio = _sum_render(lead_entries)
        bgv_audio = _sum_render(bgv_entries)
        lead_rms = float(metrics_for(mono_sum(lead_audio), sr, instrument="lead_vocal").get("rms_db", -100.0))
        bgv_rms = float(metrics_for(mono_sum(bgv_audio), sr, instrument="backing_vocal").get("rms_db", -100.0))
        lead_presence = _band_rms_db(lead_audio, sr, 2200.0, 5200.0)
        bgv_presence = _band_rms_db(bgv_audio, sr, 2200.0, 5200.0)
        return {
            "lead_over_bgv_rms_db": lead_rms - bgv_rms,
            "lead_over_bgv_presence_db": lead_presence - bgv_presence,
        }

    def _append_bus_action(
        *,
        action_type: str,
        channel: int,
        bus_id: int,
        before_db: float,
        after_db: float,
    ) -> None:
        if abs(after_db - before_db) < 0.2:
            return
        actions.append({
            "type": action_type,
            "channel": channel,
            "bus_id": bus_id,
            "before_db": round(before_db, 2),
            "after_db": round(after_db, 2),
        })

    before_metrics = _measure_gap()

    hierarchy = targets.get("hierarchy") or {}
    target_gap = float(hierarchy.get("lead_over_bgv_rms_db", 4.2))
    target_presence_gap = float(hierarchy.get("lead_over_bgv_presence_db", target_gap))
    style_width = float(targets.get("style_summary", {}).get("stereo_width", 0.18))
    base_bgv_pan_cap = float(np.clip(0.11 + style_width * 0.26, 0.11, 0.20))
    space_lift = _reference_fx_space_lift(reference_context)

    actions: list[dict[str, Any]] = []
    rounds = 0
    for _ in range(3):
        current = _measure_gap()
        excess_gap = max(
            0.0,
            current["lead_over_bgv_rms_db"] - target_gap,
            (current["lead_over_bgv_presence_db"] - target_presence_gap) * 0.88,
        )
        shortage_gap = max(0.0, target_gap - current["lead_over_bgv_rms_db"])
        if excess_gap <= 0.35 and shortage_gap <= 0.35:
            break

        rounds += 1
        if excess_gap > 0.35:
            lead_cut = float(np.clip(0.28 + excess_gap * 0.16, 0.28, 0.95))
            bgv_lift = float(np.clip(0.12 + excess_gap * 0.06, 0.0, 0.38))
            bgv_pan_cap = float(np.clip(base_bgv_pan_cap - min(0.07, excess_gap * 0.022), 0.10, base_bgv_pan_cap))
            for channel, plan in lead_entries:
                before = float(plan.fader_db)
                plan.fader_db = float(np.clip(plan.fader_db - lead_cut, -30.0, 0.0))
                if plan.fader_db != before:
                    actions.append({
                        "type": "lead_trim_to_reference_space",
                        "channel": channel,
                        "before_db": round(before, 2),
                        "after_db": round(float(plan.fader_db), 2),
                    })
                lead_plate = float(plan.fx_bus_send_db.get(13, plan.fx_send_db if plan.fx_send_db is not None else -18.0))
                lead_delay = float(plan.fx_bus_send_db.get(15, lead_plate - 6.0))
                after_plate = float(np.clip(lead_plate + (0.18 + excess_gap * 0.08 + space_lift * 0.14), -96.0, 0.0))
                after_delay = float(np.clip(lead_delay + (0.45 + excess_gap * 0.12 + space_lift * 0.25), -96.0, 0.0))
                plan.fx_bus_send_db[13] = after_plate
                plan.fx_bus_send_db[15] = after_delay
                _append_bus_action(
                    action_type="lead_fx_send_wet_to_reference_space",
                    channel=channel,
                    bus_id=13,
                    before_db=lead_plate,
                    after_db=after_plate,
                )
                _append_bus_action(
                    action_type="lead_fx_send_wet_to_reference_space",
                    channel=channel,
                    bus_id=15,
                    before_db=lead_delay,
                    after_db=after_delay,
                )
            for idx, (channel, plan) in enumerate(bgv_entries):
                before = float(plan.fader_db)
                plan.fader_db = float(np.clip(plan.fader_db + bgv_lift, -30.0, 0.0))
                if plan.fader_db != before:
                    actions.append({
                        "type": "bgv_lift_to_reference_space",
                        "channel": channel,
                        "before_db": round(before, 2),
                        "after_db": round(float(plan.fader_db), 2),
                    })
                before_pan = float(plan.pan)
                sign = np.sign(plan.pan) if abs(plan.pan) > 1e-4 else (-1.0 if idx == 0 else 1.0)
                plan.pan = float(sign * min(abs(plan.pan) if abs(plan.pan) > 1e-4 else bgv_pan_cap, bgv_pan_cap))
                if abs(plan.pan - before_pan) >= 0.01:
                    actions.append({
                        "type": "bgv_pan_narrow_to_reference_space",
                        "channel": channel,
                        "before": round(before_pan, 3),
                        "after": round(float(plan.pan), 3),
                    })
                bgv_plate = float(plan.fx_bus_send_db.get(13, plan.fx_send_db if plan.fx_send_db is not None else -16.0))
                bgv_doubler = float(plan.fx_bus_send_db.get(16, bgv_plate - 8.0))
                after_plate = float(np.clip(bgv_plate + (0.25 + excess_gap * 0.09 + space_lift * 0.18), -96.0, 0.0))
                after_doubler = float(np.clip(bgv_doubler + (0.7 + excess_gap * 0.16 + space_lift * 0.30), -96.0, 0.0))
                plan.fx_bus_send_db[13] = after_plate
                plan.fx_bus_send_db[16] = after_doubler
                _append_bus_action(
                    action_type="bgv_fx_send_wet_to_reference_space",
                    channel=channel,
                    bus_id=13,
                    before_db=bgv_plate,
                    after_db=after_plate,
                )
                _append_bus_action(
                    action_type="bgv_fx_send_wet_to_reference_space",
                    channel=channel,
                    bus_id=16,
                    before_db=bgv_doubler,
                    after_db=after_doubler,
                )
            continue

        lead_lift = float(np.clip(0.2 + shortage_gap * 0.10, 0.2, 0.7))
        bgv_trim = float(np.clip(0.2 + shortage_gap * 0.08, 0.2, 0.6))
        for channel, plan in lead_entries:
            before = float(plan.fader_db)
            plan.fader_db = float(np.clip(plan.fader_db + lead_lift, -30.0, 0.0))
            if plan.fader_db != before:
                actions.append({
                    "type": "lead_lift_to_reference_space",
                    "channel": channel,
                    "before_db": round(before, 2),
                    "after_db": round(float(plan.fader_db), 2),
                })
        for channel, plan in bgv_entries:
            before = float(plan.fader_db)
            plan.fader_db = float(np.clip(plan.fader_db - bgv_trim, -30.0, 0.0))
            if plan.fader_db != before:
                actions.append({
                    "type": "bgv_trim_to_reference_space",
                    "channel": channel,
                    "before_db": round(before, 2),
                    "after_db": round(float(plan.fader_db), 2),
                })

    after_metrics = _measure_gap()

    return {
        "enabled": True,
        "applied": bool(actions),
        "rounds": rounds,
        "target_lead_over_bgv_rms_db": round(target_gap, 3),
        "target_lead_over_bgv_presence_db": round(target_presence_gap, 3),
        "analysis_preview_sec": round(float(analysis_preview_sec), 2),
        "before": {
            "lead_over_bgv_rms_db": round(before_metrics["lead_over_bgv_rms_db"], 3),
            "lead_over_bgv_presence_db": round(before_metrics["lead_over_bgv_presence_db"], 3),
        },
        "after": {
            "lead_over_bgv_rms_db": round(after_metrics["lead_over_bgv_rms_db"], 3),
            "lead_over_bgv_presence_db": round(after_metrics["lead_over_bgv_presence_db"], 3),
        },
        "actions": actions,
        "notes": [
            "This focused pass only adjusts lead, backing vocals, and their FX space.",
            "Low end, kick/bass hierarchy, cymbals, and drum balance stay untouched in this pass.",
        ],
    }


def apply_cymbal_cleanup_for_kick_focus(plans: dict[int, ChannelPlan], cymbal_atten_db: float) -> dict[str, Any]:
    """Reduce cymbal stems slightly so kick body is easier to hear."""
    if cymbal_atten_db <= 0.0:
        return {"enabled": False}

    target_instruments = {
        "hi_hat",
        "ride",
    }
    changes = []
    for channel, plan in plans.items():
        if plan.muted or plan.instrument not in target_instruments:
            continue
        before = plan.fader_db
        after = float(np.clip(before - cymbal_atten_db, -100.0, 10.0))
        if after == before:
            continue
        plan.fader_db = after
        changes.append({
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "fader_before_db": round(before, 2),
            "fader_after_db": round(after, 2),
            "atten_db": round(cymbal_atten_db, 2),
        })

    return {
        "enabled": bool(changes),
        "atten_db": round(cymbal_atten_db, 2),
        "changes": changes,
    }


def _filter_stereo_return(audio: np.ndarray, sr: int, hpf_hz: float, lpf_hz: float) -> np.ndarray:
    out = np.asarray(audio, dtype=np.float32).copy()
    for ch in range(2):
        out[:, ch] = highpass(out[:, ch], sr, hpf_hz)
        out[:, ch] = lowpass(out[:, ch], sr, lpf_hz)
    return out.astype(np.float32)


def _simple_reverb(bus_input: np.ndarray, sr: int, bus: FXBusDecision) -> np.ndarray:
    predelay = int(float(bus.params.get("predelay_ms", 30.0)) * sr / 1000.0)
    decay_s = max(0.2, float(bus.params.get("decay_s", 1.0)))
    density = float(bus.params.get("density", 0.7))
    brightness = float(bus.params.get("brightness", 0.5))
    wet = np.zeros_like(bus_input, dtype=np.float32)
    source = np.pad(bus_input, ((predelay, 0), (0, 0)))[:len(bus_input)]

    # Sparse multi-tap reverb: intentionally lightweight for full-song offline
    # renders while still producing early reflections and a short musical tail.
    early_ms = [19, 29, 37, 43, 53, 67, 79, 97]
    tail_ms = np.linspace(115.0, max(170.0, decay_s * 820.0), int(10 + density * 12))
    all_taps = list(early_ms) + [float(v) for v in tail_ms]
    for idx, delay_ms_value in enumerate(all_taps):
        delay = max(1, int(delay_ms_value * sr / 1000.0))
        if delay >= len(source):
            continue
        time_s = delay / sr
        gain = math.exp(-time_s / max(0.15, decay_s * 0.52)) * (0.30 if idx < len(early_ms) else 0.18)
        gain *= 1.0 + (density - 0.5) * 0.18
        if idx % 2 == 0:
            wet[delay:, 0] += source[:-delay, 0] * gain
            wet[delay:, 1] += source[:-delay, 1] * gain * 0.82
        else:
            wet[delay:, 0] += source[:-delay, 1] * gain * 0.82
            wet[delay:, 1] += source[:-delay, 0] * gain

    wet *= 1.35
    wet = _filter_stereo_return(wet, sr, bus.hpf_hz, min(bus.lpf_hz, 4200.0 + brightness * 5200.0))
    return wet.astype(np.float32)


def _tempo_delay(bus_input: np.ndarray, sr: int, bus: FXBusDecision) -> np.ndarray:
    left_delay = int(float(bus.params.get("left_delay_ms", 375.0)) * sr / 1000.0)
    right_delay = int(float(bus.params.get("right_delay_ms", 500.0)) * sr / 1000.0)
    feedback = float(np.clip(bus.params.get("feedback", 0.2), 0.0, 0.55))
    wet = np.zeros_like(bus_input, dtype=np.float32)
    mono = mono_sum(bus_input)
    for repeat in range(1, 4):
        gain = feedback ** (repeat - 1)
        l_start = left_delay * repeat
        r_start = right_delay * repeat
        if l_start < len(wet):
            wet[l_start:, 0] += mono[:len(wet) - l_start] * gain
        if r_start < len(wet):
            wet[r_start:, 1] += mono[:len(wet) - r_start] * gain
    wet *= 1.0
    return _filter_stereo_return(wet, sr, bus.hpf_hz, bus.lpf_hz)


def _chorus_doubler(bus_input: np.ndarray, sr: int, bus: FXBusDecision) -> np.ndarray:
    left_delay = int(float(bus.params.get("left_delay_ms", 11.0)) * sr / 1000.0)
    right_delay = int(float(bus.params.get("right_delay_ms", 17.0)) * sr / 1000.0)
    depth = float(np.clip(bus.params.get("depth", 0.16), 0.0, 0.5))
    mono = mono_sum(bus_input)
    wet = np.zeros_like(bus_input, dtype=np.float32)
    if left_delay < len(wet):
        wet[left_delay:, 0] += mono[:len(wet) - left_delay]
    if right_delay < len(wet):
        wet[right_delay:, 1] += mono[:len(wet) - right_delay]
    # A second quiet tap mimics chorus spread without a phase-heavy modulated insert.
    second_l = left_delay + int(7.0 * sr / 1000.0)
    second_r = right_delay + int(9.0 * sr / 1000.0)
    if second_l < len(wet):
        wet[second_l:, 0] -= mono[:len(wet) - second_l] * 0.35
    if second_r < len(wet):
        wet[second_r:, 1] -= mono[:len(wet) - second_r] * 0.35
    wet *= depth * 2.4
    return _filter_stereo_return(wet, sr, bus.hpf_hz, bus.lpf_hz)


def _duck_fx_return(
    fx_return: np.ndarray,
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    source_instrument: str,
    depth_db: float,
) -> np.ndarray:
    if depth_db <= 0.0:
        return fx_return
    source_channels = [
        channel for channel, plan in plans.items()
        if plan.instrument == source_instrument and channel in rendered_channels and not plan.muted
    ]
    if not source_channels:
        return fx_return
    source = sum((rendered_channels[channel] for channel in source_channels), np.zeros_like(fx_return))
    frame = int(0.22 * sr)
    hop = int(0.05 * sr)
    starts, source_db = _frame_rms_db(source, frame, hop)
    active = np.clip((source_db + 36.0) / 12.0, 0.0, 1.0)
    duck_db = _smooth_gain_db((-depth_db * active).astype(np.float32), sr, hop, attack_ms=90.0, release_ms=650.0)
    sample_points = np.clip(starts + frame // 2, 0, len(fx_return) - 1)
    full_points = np.concatenate(([0.0], sample_points, [float(len(fx_return) - 1)]))
    full_duck = np.concatenate(([duck_db[0]], duck_db, [duck_db[-1]]))
    envelope_db = np.interp(np.arange(len(fx_return)), full_points, full_duck).astype(np.float32)
    return (fx_return * (10.0 ** (envelope_db / 20.0))[:, None]).astype(np.float32)


def apply_offline_fx_plan(
    rendered_channels: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    tempo_bpm: float = 120.0,
    reference_context: ReferenceMixContext | None = None,
    source_layer: SourceKnowledgeLayer | None = None,
    source_session_id: str = "",
    fx_plan_override: FXPlan | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Render shared stereo FX returns from the rule-based FX plan."""
    if fx_plan_override is not None:
        fx_plan = fx_plan_override
    else:
        planner = AutoFXPlanner(tempo_bpm=tempo_bpm, vocal_priority=True)
        fx_plan = planner.create_plan({
            channel: plan.instrument
            for channel, plan in plans.items()
            if channel in rendered_channels and not plan.muted
        })
    reference_fx_overrides: dict[str, Any] = {
        "enabled": False,
        "bus_adjustments": [],
        "send_adjustments": [],
    }
    if reference_context is not None:
        style = reference_context.style_profile
        width = float(np.clip(style.stereo_width, 0.0, 1.0))
        narrowness = float(np.clip(0.28 - width, 0.0, 0.28) / 0.28)
        tightness = float(np.clip(8.0 - float(style.dynamic_range), 0.0, 8.0) / 8.0)
        space_lift = _reference_fx_space_lift(reference_context)
        darkness = float(np.clip(
            (float(style.spectral_balance.get("mid", -11.5)) - float(style.spectral_balance.get("presence", -19.5))) / 10.0,
            0.0,
            1.4,
        ))
        buses = []
        for bus in fx_plan.buses:
            params = dict(bus.params)
            return_level_db = float(bus.return_level_db)
            lpf_hz = float(bus.lpf_hz)
            duck_depth_db = float(bus.duck_depth_db)
            changed = {}
            if bus.fx_type == "reverb":
                return_level_db += 0.2 + space_lift * 0.75 - narrowness * 0.25
                decay_scale = float(np.clip(0.92 + space_lift * 0.10 - tightness * 0.04, 0.78, 1.15))
                params["decay_s"] = max(0.45, float(params.get("decay_s", 1.0)) * decay_scale)
                params["predelay_ms"] = float(params.get("predelay_ms", 30.0)) * (0.96 + width * 0.14)
                params["brightness"] = float(np.clip(float(params.get("brightness", 0.5)) - darkness * 0.12 - narrowness * 0.05 + space_lift * 0.04, 0.12, 0.78))
                lpf_hz = float(np.clip(lpf_hz - 700.0 * darkness - 450.0 * narrowness, 3400.0, bus.lpf_hz))
                duck_depth_db = max(0.0, duck_depth_db - (0.2 + space_lift * 0.25))
                changed = {
                    "return_before_db": round(float(bus.return_level_db), 2),
                    "return_after_db": round(return_level_db, 2),
                    "decay_after_s": round(float(params["decay_s"]), 3),
                    "brightness_after": round(float(params["brightness"]), 3),
                    "lpf_after_hz": round(lpf_hz, 1),
                    "duck_after_db": round(float(duck_depth_db), 3),
                }
            elif bus.fx_type == "delay":
                return_level_db += 0.35 + space_lift * 1.0 - narrowness * 0.35
                feedback_scale = float(np.clip(0.88 + space_lift * 0.12 - tightness * 0.05, 0.82, 1.18))
                params["feedback"] = float(np.clip(float(params.get("feedback", 0.2)) * feedback_scale, 0.10, 0.38))
                params["width"] = float(np.clip(float(params.get("width", 0.8)) * (0.78 + width * 0.28 + space_lift * 0.10), 0.35, 0.95))
                lpf_hz = float(np.clip(lpf_hz - 550.0 * darkness - 650.0 * narrowness, 2800.0, bus.lpf_hz))
                duck_depth_db = max(0.0, duck_depth_db - (0.45 + space_lift * 0.35))
                changed = {
                    "return_before_db": round(float(bus.return_level_db), 2),
                    "return_after_db": round(return_level_db, 2),
                    "feedback_after": round(float(params["feedback"]), 3),
                    "width_after": round(float(params["width"]), 3),
                    "lpf_after_hz": round(lpf_hz, 1),
                    "duck_after_db": round(float(duck_depth_db), 3),
                }
            elif bus.fx_type == "chorus":
                return_level_db += 0.5 + space_lift * 1.3 - narrowness * 0.9
                params["depth"] = float(np.clip(float(params.get("depth", 0.16)) * (0.74 + width * 0.42 + space_lift * 0.16), 0.08, 0.22))
                params["left_delay_ms"] = float(params.get("left_delay_ms", 11.0)) * (0.9 + width * 0.08)
                params["right_delay_ms"] = float(params.get("right_delay_ms", 17.0)) * (0.9 + width * 0.08)
                lpf_hz = float(np.clip(lpf_hz - 800.0 * darkness - 800.0 * narrowness, 3200.0, bus.lpf_hz))
                changed = {
                    "return_before_db": round(float(bus.return_level_db), 2),
                    "return_after_db": round(return_level_db, 2),
                    "depth_after": round(float(params["depth"]), 3),
                    "lpf_after_hz": round(lpf_hz, 1),
                }
            new_bus = replace(
                bus,
                return_level_db=float(return_level_db),
                params=params,
                lpf_hz=float(lpf_hz),
                duck_depth_db=float(duck_depth_db),
            )
            buses.append(new_bus)
            if changed:
                reference_fx_overrides["bus_adjustments"].append({
                    "bus_id": bus.bus_id,
                    "name": bus.name,
                    "fx_type": bus.fx_type,
                    **changed,
                })

        sends = []
        for send in fx_plan.sends:
            send_db = float(send.send_db)
            before = send_db
            if send.bus_id == 16:
                send_db += 0.7 + space_lift * 1.15 - narrowness * 0.9
            elif send.bus_id == 15:
                send_db += 0.45 + space_lift * 1.05 - narrowness * 0.55
            elif send.bus_id == 13:
                send_db += 0.3 + space_lift * 0.7 - narrowness * 0.35
            elif send.bus_id == 14:
                send_db += 0.2 + space_lift * 0.55 - narrowness * 0.3
            send_db = float(np.clip(send_db, -96.0, 12.0))
            new_send = replace(send, send_db=send_db)
            sends.append(new_send)
            if abs(send_db - before) >= 0.1:
                reference_fx_overrides["send_adjustments"].append({
                    "channel": send.channel_id,
                    "instrument": send.instrument,
                    "bus_id": send.bus_id,
                    "before_db": round(before, 2),
                    "after_db": round(send_db, 2),
                })
        fx_plan = FXPlan(buses=buses, sends=sends, notes=fx_plan.notes)
        reference_fx_overrides["enabled"] = bool(reference_fx_overrides["bus_adjustments"] or reference_fx_overrides["send_adjustments"])
        reference_fx_overrides["style_summary"] = {
            "stereo_width": round(width, 3),
            "dynamic_range_db": round(float(style.dynamic_range), 3),
            "narrowness": round(narrowness, 3),
            "tightness": round(tightness, 3),
            "darkness": round(darkness, 3),
            "space_lift": round(space_lift, 3),
        }

    adjusted_sends = []
    if fx_plan.sends:
        sends = []
        for send in fx_plan.sends:
            plan = plans.get(send.channel_id)
            if plan is None:
                sends.append(send)
                continue
            raw_target = plan.fx_bus_send_db.get(send.bus_id)
            if raw_target is None:
                raw_target = plan.fx_send_db
            if raw_target is None:
                sends.append(send)
                continue
            target_send = float(np.clip(raw_target, -96.0, 0.0))
            blended_send = float(np.clip(send.send_db + (target_send - send.send_db) * 0.55, -96.0, 12.0))
            if abs(blended_send - send.send_db) < 0.1:
                sends.append(send)
                continue
            sends.append(replace(send, send_db=blended_send))
            adjusted_sends.append({
                "channel": send.channel_id,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "before_db": round(float(send.send_db), 2),
                "target_db": round(target_send, 2),
                "after_db": round(blended_send, 2),
            })
        fx_plan = FXPlan(buses=fx_plan.buses, sends=sends, notes=fx_plan.notes)

    target_len = len(next(iter(rendered_channels.values()))) if rendered_channels else 0
    returns: dict[str, np.ndarray] = {}
    return_reports: list[dict[str, Any]] = []
    sends_by_bus: dict[int, list[dict[str, Any]]] = {}

    for bus in fx_plan.buses:
        bus_input = np.zeros((target_len, 2), dtype=np.float32)
        active_sends = []
        for send in fx_plan.sends:
            if send.bus_id != bus.bus_id or send.channel_id not in rendered_channels:
                continue
            bus_input += rendered_channels[send.channel_id] * db_to_amp(send.send_db)
            item = send.__dict__.copy()
            item["file"] = plans[send.channel_id].path.name
            active_sends.append(item)
        sends_by_bus[bus.bus_id] = active_sends

        if not active_sends or target_len == 0:
            continue
        if bus.fx_type == "reverb":
            fx_return = _simple_reverb(bus_input, sr, bus)
        elif bus.fx_type == "delay":
            fx_return = _tempo_delay(bus_input, sr, bus)
        elif bus.fx_type == "chorus":
            fx_return = _chorus_doubler(bus_input, sr, bus)
        else:
            continue

        if bus.duck_source:
            fx_return = _duck_fx_return(fx_return, rendered_channels, plans, sr, bus.duck_source, bus.duck_depth_db)
        fx_return = (fx_return * db_to_amp(bus.return_level_db)).astype(np.float32)
        key = f"{bus.bus_id}_{bus.name.lower().replace(' ', '_')}"
        returns[key] = fx_return
        if source_layer is not None and source_layer.enabled:
            first_instrument = str(active_sends[0].get("instrument", "all")) if active_sends else "all"
            fx_domains = [str(bus.fx_type), "depth", "fx"]
            fx_problems = ["dry_vocal", "no_depth", "muddy_fx", "washed_out_front"]
            bus_matches = _safe_source_retrieve(
                source_layer,
                f"{first_instrument} {bus.fx_type} fx depth filtered return",
                domains=fx_domains,
                instrument=first_instrument,
                problems=fx_problems,
                action_types=["fx_send_candidate"],
            )
            _record_source_candidate(
                source_layer,
                session_id=source_session_id or "offline_mix",
                decision_id=_source_decision_slug(source_session_id, "fx", bus.bus_id, "return"),
                channel=f"FX:{bus.name}",
                instrument=first_instrument,
                category="fx",
                problem="no_depth",
                matches=bus_matches,
                action={
                    "action_type": "fx_candidate",
                    "candidate_kind": "return_bus",
                    "bus_id": int(bus.bus_id),
                    "name": bus.name,
                    "fx_type": bus.fx_type,
                    "return_level_db": round(float(bus.return_level_db), 3),
                    "hpf_hz": round(float(bus.hpf_hz), 3),
                    "lpf_hz": round(float(bus.lpf_hz), 3),
                    "duck_source": bus.duck_source,
                    "duck_depth_db": round(float(bus.duck_depth_db), 3),
                    "active_send_count": len(active_sends),
                    "params": dict(bus.params),
                },
                before_audio=bus_input,
                after_audio=fx_return,
                sr=sr,
                context={"stage": "fx_return", "active_sends": active_sends},
            )
            for send_item in active_sends:
                send_channel = int(send_item.get("channel_id", 0) or 0)
                send_instrument = str(send_item.get("instrument", "all"))
                if send_channel not in rendered_channels:
                    continue
                send_problem = "dry_vocal" if "vocal" in send_instrument else "no_depth"
                send_matches = _safe_source_retrieve(
                    source_layer,
                    f"{send_instrument} {bus.fx_type} send depth filtered return",
                    domains=fx_domains,
                    instrument=send_instrument,
                    problems=[send_problem, "no_depth", "muddy_fx"],
                    action_types=["fx_send_candidate"],
                )
                send_audio = rendered_channels[send_channel] * db_to_amp(float(send_item["send_db"]))
                _record_source_candidate(
                    source_layer,
                    session_id=source_session_id or "offline_mix",
                    decision_id=_source_decision_slug(
                        source_session_id,
                        "fx",
                        bus.bus_id,
                        "send",
                        send_channel,
                    ),
                    channel=f"{send_channel}:{send_item.get('file', '')}",
                    instrument=send_instrument,
                    category="fx",
                    problem=send_problem,
                    matches=send_matches,
                    action={
                        "action_type": "fx_send_candidate",
                        "candidate_kind": "send_to_return",
                        "bus_id": int(bus.bus_id),
                        "bus_name": bus.name,
                        "fx_type": bus.fx_type,
                        "send_db": round(float(send_item["send_db"]), 3),
                        "post_fader": bool(send_item.get("post_fader", True)),
                        "reason": str(send_item.get("reason", "")),
                    },
                    before_audio=send_audio,
                    after_audio=fx_return,
                    sr=sr,
                    context={"stage": "fx_send", "bus": bus.__dict__},
                )
        return_reports.append({
            **bus.__dict__,
            "active_send_count": len(active_sends),
            "return_peak_dbfs": round(amp_to_db(float(np.max(np.abs(fx_return))) if len(fx_return) else 0.0), 2),
        })

    return returns, {
        "enabled": True,
        "tempo_bpm": tempo_bpm,
        "plan_override": bool(fx_plan_override is not None),
        "plan": fx_plan.to_dict(),
        "returns": return_reports,
        "sends_by_bus": sends_by_bus,
        "reference_send_overrides": adjusted_sends,
        "reference_fx_overrides": reference_fx_overrides,
    }


def _mid_side_levels(audio: np.ndarray) -> dict[str, float]:
    if len(audio) == 0:
        return {"mid_db": -120.0, "side_db": -120.0, "side_minus_mid_db": 0.0, "correlation": 1.0}
    left = np.asarray(audio[:, 0], dtype=np.float32)
    right = np.asarray(audio[:, 1], dtype=np.float32)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_rms = float(np.sqrt(np.mean(np.square(mid))) + 1e-12)
    side_rms = float(np.sqrt(np.mean(np.square(side))) + 1e-12)
    corr = float(np.corrcoef(left, right)[0, 1]) if len(left) > 1 else 1.0
    return {
        "mid_db": round(amp_to_db(mid_rms), 2),
        "side_db": round(amp_to_db(side_rms), 2),
        "side_minus_mid_db": round(amp_to_db(side_rms / max(mid_rms, 1e-12)), 2),
        "correlation": round(float(np.clip(corr, -1.0, 1.0)), 4),
    }


def _apply_stereo_width_polish(
    audio: np.ndarray,
    sr: int,
    *,
    width_gain: float,
    side_hpf_hz: float = 180.0,
    side_air_boost_db: float = 0.0,
) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    left = np.asarray(audio[:, 0], dtype=np.float32)
    right = np.asarray(audio[:, 1], dtype=np.float32)
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    side_low = _lowpass_zero_phase(side, sr, side_hpf_hz)
    side_high = side - side_low
    polished_side = side_low + side_high * float(max(width_gain, 1.0))
    if side_air_boost_db > 0.05:
        polished_side = peaking_eq(polished_side, sr, 9800.0, side_air_boost_db, 0.72)
    out = np.column_stack((mid + polished_side, mid - polished_side)).astype(np.float32)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return out


def _compress_band_component(
    audio: np.ndarray,
    sr: int,
    *,
    low_hz: float,
    high_hz: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    threshold_offset_db: float = 0.5,
) -> np.ndarray:
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) == 0:
        return np.asarray(audio, dtype=np.float32)
    out = np.asarray(audio, dtype=np.float32).copy()
    for ch in range(2):
        lane = out[:, ch]
        band = _bandpass_zero_phase(lane, sr, low_hz, high_hz)
        band_rms = float(np.sqrt(np.mean(np.square(band))) + 1e-12)
        threshold_db = amp_to_db(band_rms) + threshold_offset_db
        compressed_band = compressor(
            band,
            sr,
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            makeup_db=0.0,
            auto_makeup=False,
        )
        out[:, ch] = (lane - band + compressed_band).astype(np.float32)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return out


def apply_large_system_translation_polish(
    rendered_channels: dict[int, np.ndarray],
    fx_returns: dict[int, np.ndarray],
    plans: dict[int, ChannelPlan],
    sr: int,
    *,
    reference_context: ReferenceMixContext | None = None,
    genre: str | None = None,
) -> dict[str, Any]:
    if not rendered_channels:
        return {"enabled": False, "reason": "no_rendered_channels"}

    targets = _effective_balance_targets(reference_context, genre=genre)
    style_width = float((targets.get("style_summary") or {}).get("stereo_width", 0.42))

    def _rendered_mix() -> np.ndarray:
        first = next(iter(rendered_channels.values()))
        mix = np.zeros_like(first)
        for audio in rendered_channels.values():
            mix += audio
        for audio in fx_returns.values():
            mix += audio
        return mix.astype(np.float32)

    mix_before = _rendered_mix()
    before_width = _mid_side_levels(mix_before)
    actions: list[dict[str, Any]] = []

    low_end_candidates: list[tuple[float, int, float, float]] = []
    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        if plan.instrument not in (BASS_INSTRUMENTS | {"playback", "accordion"}):
            continue
        dominant_db = _band_rms_db(audio, sr, 55.0, 70.0)
        punch_db = _band_rms_db(audio, sr, 90.0, 120.0)
        delta_db = dominant_db - punch_db
        if dominant_db > -42.0 and delta_db > 2.4:
            low_end_candidates.append((delta_db, channel, dominant_db, punch_db))

    for delta_db, channel, dominant_db, punch_db in sorted(low_end_candidates, reverse=True)[:3]:
        plan = plans[channel]
        before = rendered_channels[channel]
        ratio = float(np.clip(1.65 + (delta_db - 2.4) * 0.18, 1.65, 2.35))
        after = _compress_band_component(
            before,
            sr,
            low_hz=55.0,
            high_hz=70.0,
            ratio=ratio,
            attack_ms=42.0,
            release_ms=240.0,
            threshold_offset_db=0.45,
        )
        if plan.instrument in {"bass", "bass_guitar", "bass_di", "bass_mic", "synth_bass"}:
            for idx in range(2):
                after[:, idx] = peaking_eq(after[:, idx], sr, 62.0, -0.6, 0.95)
        rendered_channels[channel] = after.astype(np.float32)
        actions.append({
            "type": "low_end_band_control",
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "band_hz": "55-70",
            "dominant_before_db": round(dominant_db, 2),
            "punch_before_db": round(punch_db, 2),
            "ratio": round(ratio, 2),
            "reason": "Large-system polish kept the 55-70 Hz dominance under control so the PA does not over-bloom around the main bass resonance.",
        })

    low_mid_candidates: list[tuple[float, int, float, float]] = []
    for channel, audio in rendered_channels.items():
        plan = plans.get(channel)
        if plan is None or plan.muted:
            continue
        if plan.instrument not in (WINDOW_SPACE_COMPETITOR_INSTRUMENTS | {"backing_vocal"}):
            continue
        low_mid_db = _band_rms_db(audio, sr, 180.0, 350.0)
        focus_db = _band_rms_db(audio, sr, 700.0, 1500.0)
        delta_db = low_mid_db - focus_db
        if low_mid_db > -40.0 and delta_db > 1.8:
            low_mid_candidates.append((delta_db, channel, low_mid_db, focus_db))

    for delta_db, channel, low_mid_db, focus_db in sorted(low_mid_candidates, reverse=True)[:4]:
        plan = plans[channel]
        cut_db = float(np.clip(0.7 + (delta_db - 1.8) * 0.18, 0.7, 1.55))
        before = rendered_channels[channel]
        after = before.copy()
        for idx in range(2):
            after[:, idx] = peaking_eq(after[:, idx], sr, 265.0, -cut_db, 1.08)
        rendered_channels[channel] = after.astype(np.float32)
        actions.append({
            "type": "low_mid_cleanup",
            "channel": channel,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "frequency_hz": 265.0,
            "gain_db": round(-cut_db, 2),
            "low_mid_before_db": round(low_mid_db, 2),
            "focus_before_db": round(focus_db, 2),
            "reason": "Large-system polish cleaned the 180-350 Hz body build-up so the center translates with less boxiness on PA.",
        })

    for bus_id, audio in list(fx_returns.items()):
        low_mid_db = _band_rms_db(audio, sr, 180.0, 350.0)
        focus_db = _band_rms_db(audio, sr, 2200.0, 5200.0)
        if low_mid_db > -42.0 and low_mid_db - focus_db > 1.4:
            after = audio.copy()
            for idx in range(2):
                after[:, idx] = peaking_eq(after[:, idx], sr, 280.0, -0.85, 1.0)
            fx_returns[bus_id] = after.astype(np.float32)
            actions.append({
                "type": "fx_low_mid_cleanup",
                "bus_id": str(bus_id),
                "frequency_hz": 280.0,
                "gain_db": -0.85,
                "reason": "Large-system polish trimmed muddy low-mid from the return so ambience sits around the lead instead of masking the center.",
            })

    width_target_db = -17.4 if style_width >= 0.46 else -18.2
    side_shortage_db = max(0.0, width_target_db - float(before_width["side_minus_mid_db"]))
    if side_shortage_db > 0.8:
        width_gain = float(np.clip(1.08 + side_shortage_db * 0.05, 1.08, 1.32))
        side_air_boost_db = float(np.clip(0.25 + side_shortage_db * 0.12, 0.25, 0.95))
        for channel, audio in list(rendered_channels.items()):
            plan = plans.get(channel)
            if plan is None or plan.muted:
                continue
            if plan.instrument not in {"playback", "electric_guitar", "accordion"}:
                continue
            if abs(float(plan.pan)) < 0.08:
                continue
            rendered_channels[channel] = _apply_stereo_width_polish(
                audio,
                sr,
                width_gain=width_gain,
                side_hpf_hz=185.0,
                side_air_boost_db=side_air_boost_db * (0.9 if plan.instrument == "accordion" else 1.0),
            )
            actions.append({
                "type": "support_width_expansion",
                "channel": channel,
                "file": plan.path.name,
                "instrument": plan.instrument,
                "width_gain": round(width_gain, 3),
                "side_air_boost_db": round(side_air_boost_db, 2),
                "reason": "Large-system polish widened support layers above the low-mid so the mix opens sideways without loosening the mono low end.",
            })

        for bus_id, audio in list(fx_returns.items()):
            fx_returns[bus_id] = _apply_stereo_width_polish(
                audio,
                sr,
                width_gain=min(1.4, width_gain + 0.08),
                side_hpf_hz=220.0,
                side_air_boost_db=min(1.1, side_air_boost_db + 0.2),
            )
            actions.append({
                "type": "fx_width_expansion",
                "bus_id": str(bus_id),
                "width_gain": round(min(1.4, width_gain + 0.08), 3),
                "reason": "Large-system polish moved ambience wider than the center image so the vocal can stay stable while the space opens around it.",
            })

    mix_after = _rendered_mix()
    return {
        "enabled": True,
        "applied": bool(actions),
        "actions": actions,
        "before": {
            "stereo": before_width,
            "band_rms_db": {
                "55_70": round(_band_rms_db(mix_before, sr, 55.0, 70.0), 2),
                "90_120": round(_band_rms_db(mix_before, sr, 90.0, 120.0), 2),
                "180_350": round(_band_rms_db(mix_before, sr, 180.0, 350.0), 2),
                "2200_4000": round(_band_rms_db(mix_before, sr, 2200.0, 4000.0), 2),
                "8000_12000": round(_band_rms_db(mix_before, sr, 8000.0, 12000.0), 2),
            },
        },
        "after": {
            "stereo": _mid_side_levels(mix_after),
            "band_rms_db": {
                "55_70": round(_band_rms_db(mix_after, sr, 55.0, 70.0), 2),
                "90_120": round(_band_rms_db(mix_after, sr, 90.0, 120.0), 2),
                "180_350": round(_band_rms_db(mix_after, sr, 180.0, 350.0), 2),
                "2200_4000": round(_band_rms_db(mix_after, sr, 2200.0, 4000.0), 2),
                "8000_12000": round(_band_rms_db(mix_after, sr, 8000.0, 12000.0), 2),
            },
        },
        "notes": [
            "Large-system polish treats the mix like a playback master for PA: control the 55-70 Hz dominance, declutter 180-350 Hz, and create width around the center instead of inside the sub range.",
            "Any stereo expansion is applied only above the side high-pass so the low end stays mono-stable.",
        ],
    }


def _soft_limiter(mix: np.ndarray, drive_db: float, ceiling_db: float = -1.0) -> np.ndarray:
    drive = db_to_amp(drive_db)
    ceiling = db_to_amp(ceiling_db)
    shaped = np.tanh(mix * drive) / np.tanh(drive)
    peak = np.max(np.abs(shaped))
    if peak > 0:
        shaped = shaped / peak * min(ceiling, peak)
    return shaped.astype(np.float32)


def _conform_master_to_target_lufs(
    mix: np.ndarray,
    meter: pyln.Meter,
    target_lufs: float,
    ceiling_dbfs: float = -1.0,
) -> tuple[np.ndarray, float | None, bool]:
    """Bring the final stereo mix toward the requested integrated loudness target."""
    conformed = np.asarray(mix, dtype=np.float32).copy()
    ceiling = db_to_amp(ceiling_dbfs)
    soft_limiter_used = False

    peak = float(np.max(np.abs(conformed))) if len(conformed) else 0.0
    if peak > 0.95:
        conformed = conformed / peak * 0.95

    for _ in range(4):
        try:
            loudness = float(meter.integrated_loudness(conformed))
        except Exception:
            return conformed.astype(np.float32), None, soft_limiter_used
        if not np.isfinite(loudness):
            return conformed.astype(np.float32), None, soft_limiter_used

        needed = float(target_lufs - loudness)
        if abs(needed) < 0.4:
            break

        conformed = conformed * db_to_amp(min(max(needed, -6.0), 6.0))
        peak = float(np.max(np.abs(conformed))) if len(conformed) else 0.0
        if peak > ceiling:
            over_db = amp_to_db(peak / ceiling)
            conformed = _soft_limiter(
                conformed / max(peak, 1e-9),
                drive_db=min(12.0, max(3.0, over_db + 3.0)),
                ceiling_db=ceiling_dbfs,
            )
            soft_limiter_used = True

    peak = float(np.max(np.abs(conformed))) if len(conformed) else 0.0
    if peak > ceiling:
        conformed = conformed / peak * ceiling

    post_lufs = None
    try:
        post_lufs = float(meter.integrated_loudness(conformed))
    except Exception:
        pass
    return conformed.astype(np.float32), post_lufs, soft_limiter_used


def master_process(
    mix: np.ndarray,
    sr: int,
    target_lufs: float = -16.0,
    final_limiter: bool = True,
    live_peak_ceiling_db: float = -3.0,
    reference_context: ReferenceMixContext | None = None,
    allow_reference_mastering: bool = True,
    ceiling_dbfs: float = -1.0,
    apply_bus_processing: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Console-like 2-bus cleanup and glue.
    bus_threshold_db = -9.5
    bus_ratio = 1.35
    bus_attack_ms = 32.0
    bus_release_ms = 280.0
    reference_tightness = _reference_dynamics_tightness(reference_context)
    if reference_context is not None and reference_tightness > 0.02:
        bus_threshold_db -= 0.75 * reference_tightness
        bus_ratio += 0.35 * reference_tightness
        bus_attack_ms = float(np.clip(bus_attack_ms - reference_tightness * 5.0, 22.0, 32.0))
        bus_release_ms = float(np.clip(bus_release_ms - reference_tightness * 55.0, 220.0, 280.0))
    if apply_bus_processing:
        for ch in range(2):
            mix[:, ch] = highpass(mix[:, ch], sr, 28.0)
            mix[:, ch] = compressor(
                mix[:, ch],
                sr,
                threshold_db=bus_threshold_db,
                ratio=bus_ratio,
                attack_ms=bus_attack_ms,
                release_ms=bus_release_ms,
            )

    meter = pyln.Meter(sr)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    peak_db = amp_to_db(peak)
    pre_lufs = None
    try:
        pre_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass

    reference_mastering: dict[str, Any] = {
        "requested": bool(reference_context is not None),
        "enabled": False,
    }
    bus_compressor_report = {
        "enabled": bool(apply_bus_processing),
        "threshold_db": round(float(bus_threshold_db), 2),
        "ratio": round(float(bus_ratio), 2),
        "attack_ms": round(float(bus_attack_ms), 2),
        "release_ms": round(float(bus_release_ms), 2),
        "reference_tightness": round(float(reference_tightness), 3),
    }
    if not apply_bus_processing:
        bus_compressor_report["reason"] = "disabled_by_source_rules_mert_only"

    if not final_limiter:
        target_gain_db = 0.0
        if pre_lufs is not None and np.isfinite(pre_lufs):
            target_gain_db = float(np.clip(target_lufs - pre_lufs, -6.0, 6.0))
        peak_safe_gain_db = live_peak_ceiling_db - peak_db
        static_master_gain_db = min(target_gain_db, peak_safe_gain_db)
        mix = (mix * db_to_amp(static_master_gain_db)).astype(np.float32)
        post_lufs = None
        try:
            post_lufs = float(meter.integrated_loudness(mix))
        except Exception:
            pass
        if reference_context is not None:
            reference_mastering.update({
                "reason": "final_limiter_disabled",
                "reference_path": str(reference_context.path),
                "source_type": reference_context.source_type,
                "reference_sources": [str(path) for path in reference_context.source_paths],
            })
        return np.asarray(mix, dtype=np.float32), {
            "final_limiter": False,
            "soft_limiter": False,
            "static_master_gain_db": round(static_master_gain_db, 2),
            "live_peak_ceiling_dbfs": round(live_peak_ceiling_db, 2),
            "pre_master_peak_dbfs": round(peak_db, 2),
            "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
            "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
            "bus_compressor": bus_compressor_report,
            "reference_mastering": reference_mastering,
            "note": "No final limiting or clipping stage; only static master trim is used for live-style headroom.",
        }

    if reference_context is not None and allow_reference_mastering:
        reference_input_mix = np.asarray(mix, dtype=np.float32).copy()
        reference_mastering.update({
            "reference_path": str(reference_context.path),
            "source_type": reference_context.source_type,
            "reference_sources": [str(path) for path in reference_context.source_paths],
        })
        if reference_context.audio is None or reference_context.sample_rate is None:
            reference_mastering["reason"] = "reference_has_no_audio_stream"
        else:
            ref_audio = reference_context.audio
            ref_sr = int(reference_context.sample_rate)
            if ref_sr != sr:
                ref_audio = resample_audio(ref_audio, ref_sr, sr)
            auto_master = AutoMaster(sample_rate=sr, target_lufs=target_lufs, true_peak_limit=ceiling_dbfs)
            try:
                mastered = auto_master.master(mix, reference=ref_audio, sample_rate=sr)
                if isinstance(mastered, MasteringResult):
                    mastered_audio = np.asarray(mastered.audio, dtype=np.float32)
                    reference_mastering.update({
                        "enabled": bool(mastered.success),
                        "backend": "matchering_result",
                        "peak_dbfs": round(float(mastered.peak_db), 2),
                        "lufs": round(float(mastered.lufs), 2),
                        "gain_applied_db": round(float(mastered.gain_applied_db), 2),
                        "limiter_reduction_db": round(float(mastered.limiter_reduction_db), 2),
                        "error": mastered.error,
                    })
                else:
                    mastered_audio = np.asarray(mastered, dtype=np.float32)
                    post_lufs = None
                    try:
                        post_lufs = float(meter.integrated_loudness(mastered_audio))
                    except Exception:
                        pass
                    reference_mastering.update({
                        "enabled": True,
                        "backend": "reference_audio_fallback",
                        "peak_dbfs": round(amp_to_db(float(np.max(np.abs(mastered_audio))) if len(mastered_audio) else 0.0), 2),
                        "lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
                    })
                post_lufs = None
                try:
                    post_lufs = float(meter.integrated_loudness(mastered_audio))
                except Exception:
                    pass
                expected_floor = None
                style_lufs = float(reference_context.style_profile.loudness_lufs)
                if post_lufs is not None and np.isfinite(post_lufs):
                    comparison_points = [style_lufs, target_lufs]
                    if pre_lufs is not None and np.isfinite(pre_lufs):
                        comparison_points.append(float(pre_lufs))
                    expected_floor = min(comparison_points) - 8.0
                if (
                    post_lufs is None
                    or not np.isfinite(post_lufs)
                    or (expected_floor is not None and post_lufs < expected_floor)
                ):
                    reference_mastering.update({
                        "enabled": False,
                        "reason": "reference_mastering_rejected_low_loudness",
                        "rejected_post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
                        "expected_floor_lufs": round(expected_floor, 2) if expected_floor is not None else None,
                    })
                    mix = reference_input_mix
                else:
                    reference_post_lufs = post_lufs
                    mix, post_lufs, soft_limiter_used = _conform_master_to_target_lufs(
                        mastered_audio,
                        meter,
                        target_lufs,
                        ceiling_dbfs=ceiling_dbfs,
                    )
                    reference_mastering.update({
                        "pre_target_conform_lufs": round(float(reference_post_lufs), 2)
                        if reference_post_lufs is not None and np.isfinite(reference_post_lufs)
                        else reference_mastering.get("lufs"),
                        "target_lufs": round(float(target_lufs), 2),
                    })
                    if post_lufs is not None and np.isfinite(post_lufs):
                        reference_mastering["lufs"] = round(float(post_lufs), 2)
                    return np.asarray(mix, dtype=np.float32), {
                        "final_limiter": True,
                        "soft_limiter": soft_limiter_used,
                        "ceiling_dbfs": round(float(ceiling_dbfs), 2),
                        "pre_master_peak_dbfs": round(peak_db, 2),
                        "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
                        "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
                        "bus_compressor": bus_compressor_report,
                        "reference_mastering": reference_mastering,
                    }
            except Exception as exc:
                reference_mastering.update({
                    "enabled": False,
                    "reason": "reference_mastering_failed",
                    "error": str(exc),
                })
    elif reference_context is not None:
        reference_mastering.update({
            "enabled": False,
            "reason": "reference_mastering_disabled_by_flag",
            "reference_path": str(reference_context.path),
            "source_type": reference_context.source_type,
            "reference_sources": [str(path) for path in reference_context.source_paths],
        })

    mix, post_lufs, soft_limiter_used = _conform_master_to_target_lufs(
        mix,
        meter,
        target_lufs,
        ceiling_dbfs=ceiling_dbfs,
    )
    return np.asarray(mix, dtype=np.float32), {
        "final_limiter": True,
        "soft_limiter": soft_limiter_used,
        "ceiling_dbfs": round(float(ceiling_dbfs), 2),
        "pre_master_peak_dbfs": round(peak_db, 2),
        "pre_master_lufs": round(pre_lufs, 2) if pre_lufs is not None and np.isfinite(pre_lufs) else None,
        "post_master_lufs": round(post_lufs, 2) if post_lufs is not None and np.isfinite(post_lufs) else None,
        "bus_compressor": bus_compressor_report,
        "reference_mastering": reference_mastering,
    }


def _orchestrator_dry_run_enabled(args: argparse.Namespace) -> bool:
    """Only enable dry-run when the explicit CLI flag is present."""
    return bool(getattr(args, "codex_orchestrator_dry_run", False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(Path.home() / "Desktop" / "MIX"))
    parser.add_argument("--output", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT.mp3"))
    parser.add_argument("--report", default=str(Path.home() / "Desktop" / "AUTO_MIX_AGENT_report.json"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-final-limiter", action="store_true")
    parser.add_argument("--no-drum-vocal-duck", action="store_true")
    parser.add_argument("--live-peak-ceiling-db", type=float, default=-3.0)
    parser.add_argument("--live-channel-peak-ceiling-db", type=float, default=-3.0)
    parser.add_argument("--master-ceiling-dbfs", type=float, default=-1.0)
    parser.add_argument("--live-input-fade-ms", type=float, default=25.0)
    parser.add_argument("--channel-input-fade-ms", type=float, default=4.0)
    parser.add_argument("--tempo-bpm", type=float, default=120.0)
    parser.add_argument("--disable-fx", action="store_true")
    parser.add_argument("--bass-drum-boost-db", type=float, default=0.0)
    parser.add_argument("--kick-presence-boost-db", type=float, default=0.0)
    parser.add_argument("--kick-focus-cymbal-cut-db", type=float, default=0.0)
    parser.add_argument("--no-autofoh-analyzer-pass", action="store_true")
    parser.add_argument("--codex-correction-pass", action="store_true")
    parser.add_argument("--codex-orchestrator", action="store_true")
    parser.add_argument("--codex-orchestrator-dry-run", action="store_true")
    parser.add_argument("--codex-orchestrator-allow-llm", action="store_true")
    parser.add_argument("--codex-orchestrator-max-actions", type=int, default=5)
    parser.add_argument("--soft-master", action="store_true")
    parser.add_argument("--master-target-lufs", type=float, default=-16.0)
    parser.add_argument("--reference", default="", help="Path to an external reference track or saved style preset JSON")
    parser.add_argument("--genre", default="", help="Optional genre focus for bounded mix voicing, for example 'rock'")
    parser.add_argument("--reference-dynamics-ride", action="store_true", help="Apply reference-guided stem loudness rides after channel rendering")
    parser.add_argument("--reference-vocal-fx-focus", action="store_true", help="Apply an extra narrow pass that only adjusts lead/BGV/fx space toward the reference")
    parser.add_argument("--no-reference-mastering", action="store_true", help="Use the reference only for channel/stem decisions and skip stereo reference-mastering on the final mix")
    parser.add_argument("--large-system-polish", action="store_true", help="Apply a final translation-oriented pass for PA playback: tame 55-70 Hz, clean 180-350 Hz, and open stereo support layers.")
    parser.add_argument("--autofoh-rounds", type=int, default=DEFAULT_AUTOFOH_ANALYZER_ROUNDS, help="Number of bounded AutoFOH analyzer rounds; use 3 for a slower deep pass.")
    parser.add_argument("--analysis-preview-sec", type=float, default=ANALYZER_RENDER_PREVIEW_SEC, help="Active-window render length used by offline analyzers.")
    parser.add_argument("--no-render-cache", action="store_true", help="Disable the offline channel render cache.")
    parser.add_argument("--render-cache-max-mb", type=float, default=DEFAULT_RENDER_CACHE_MAX_MB, help="Maximum memory used by the offline render cache.")
    parser.add_argument("--source-knowledge-enable", action="store_true", help="Enable shadow source-grounded logging for offline EQ/comp/pan/FX candidates.")
    parser.add_argument("--source-knowledge-log", default="", help="Override source-grounded JSONL log path for the offline pass.")
    parser.add_argument("--mert-enable", action="store_true", help="Enable offline perceptual shadow scoring with the MERT backend.")
    parser.add_argument("--mert-model-name", default="m-a-p/MERT-v1-95M", help="HuggingFace model name for --mert-enable.")
    parser.add_argument("--mert-local-files-only", action="store_true", help="Load the MERT model only from the local HuggingFace cache.")
    parser.add_argument("--perceptual-log", default="", help="Override perceptual/MERT JSONL log path for the offline pass.")
    parser.add_argument("--perceptual-window-sec", type=float, default=8.0, help="Audio window length used by offline perceptual/MERT scoring.")
    parser.add_argument("--source-rules-mert-only", action="store_true", help="Experimental offline render: build DSP only from measured metrics, rules.jsonl, and MERT shadow scoring; skip built-in project mix passes.")
    args = parser.parse_args()
    if args.source_rules_mert_only:
        args.source_knowledge_enable = True
        args.mert_enable = True
        args.no_autofoh_analyzer_pass = True
        args.codex_orchestrator = False
        args.codex_correction_pass = False
        args.reference_dynamics_ride = False
        args.reference_vocal_fx_focus = False
        args.large_system_polish = False
        args.bass_drum_boost_db = 0.0
        args.kick_presence_boost_db = 0.0
        args.kick_focus_cymbal_cut_db = 0.0

    input_dir = Path(args.input_dir)
    wavs = sorted(p for p in input_dir.glob("*.wav") if p.is_file())
    if not wavs:
        raise SystemExit(f"No WAV files found in {input_dir}")

    info = sf.info(str(wavs[0]))
    sr = int(info.samplerate)
    target_len = int(info.frames)
    plans: dict[int, ChannelPlan] = {}

    for idx, path in enumerate(wavs, start=1):
        instrument, pan, hpf, target_rms, eq, comp, phase = classify_track(path)
        mono, file_sr = read_mono(path)
        if file_sr != sr:
            raise ValueError(f"{path.name}: sample rate mismatch {file_sr} != {sr}")
        m = metrics_for(mono, sr, instrument=instrument)
        trim, trim_analysis = compute_bleed_aware_trim(instrument, target_rms, m)
        threshold, ratio, attack, release = comp
        plan = ChannelPlan(
            path=path,
            name=path.stem,
            instrument=instrument,
            pan=pan,
            hpf=hpf,
            target_rms_db=target_rms,
            trim_db=trim,
            input_fade_ms=args.live_input_fade_ms if args.no_final_limiter else args.channel_input_fade_ms,
            eq_bands=list(eq),
            comp_threshold_db=threshold,
            comp_ratio=ratio,
            comp_attack_ms=attack,
            comp_release_ms=release,
            phase_invert=phase,
            event_activity=_event_activity_ranges(mono, sr, instrument) or {},
            metrics=m,
            trim_analysis=trim_analysis,
        )
        plans[idx] = plan
        del mono

    config = yaml.safe_load((REPO_ROOT / "config" / "automixer.yaml").read_text(encoding="utf-8"))
    ai_config = config.get("ai", {})
    autofoh_config = config.get("autofoh", {})
    source_layer, source_knowledge_report = _make_offline_source_knowledge_layer(args, config)
    perceptual_evaluator, perceptual_report = _make_offline_perceptual_evaluator(args, config)
    source_session_id = _source_decision_slug(
        "offline_mix",
        input_dir.name,
        int(time.time()),
    )
    source_knowledge_report["session_id"] = source_session_id
    source_knowledge_report["channel_candidate_logs_requested"] = 0
    perceptual_report["session_id"] = source_session_id
    if args.source_rules_mert_only:
        phase_report = {
            "enabled": False,
            "reason": "source_rules_mert_only_disables_project_phase_alignment",
        }
        drum_pan_rule = {
            "enabled": False,
            "reason": "source_rules_mert_only_disables_project_overhead_panning",
        }
        source_rules_mert_only_report = apply_source_rules_mert_only_plan(
            plans,
            sr,
            target_len,
            source_layer,
            source_session_id,
        )
    else:
        phase_report = apply_drum_phase_alignment(plans, sr)
        drum_pan_rule = apply_overhead_anchored_drum_panning(plans, sr)
        source_rules_mert_only_report = {"enabled": False}
    reference_context = None if args.source_rules_mert_only else prepare_reference_mix_context(args.reference)
    genre_reference_seed = {} if args.source_rules_mert_only else _genre_reference_seed(args.genre)
    render_cache = OfflineRenderCache(
        enabled=not args.no_render_cache,
        max_bytes=int(max(0.0, float(args.render_cache_max_mb)) * 1024 * 1024),
    )
    analysis_preview_sec = max(1.0, float(args.analysis_preview_sec))
    use_llm_in_orchestrator = bool(args.codex_orchestrator_allow_llm and args.codex_orchestrator)
    llm = None
    if (not args.source_rules_mert_only) and not args.no_llm and (not args.codex_orchestrator or use_llm_in_orchestrator):
        llm = LLMClient(
            backend=ai_config.get("llm_backend", "auto"),
            model=ai_config.get("llm_model", "gpt-5.4"),
            ollama_url=ai_config.get("ollama_url", "http://localhost:11434"),
            model_fallbacks=ai_config.get("model_fallbacks") or None,
            kimi_timeout_sec=float(ai_config.get("kimi_timeout_sec", 120)),
            kimi_cli_path=ai_config.get("kimi_cli_path") or None,
            kimi_work_dir=ai_config.get("kimi_work_dir") or None,
        )

    console = VirtualConsole(plans)
    agent_mode = AgentMode.SUGGEST if args.codex_orchestrator else AgentMode.AUTO
    agent = MixingAgent(
        knowledge_base=KnowledgeBase(
            knowledge_dir=ai_config.get("knowledge_dir") or None,
            use_vector_db=False,
            allowed_categories=KnowledgeBase.AGENT_RUNTIME_CATEGORIES,
        ),
        rule_engine=RuleEngine(),
        llm_client=llm,
        mixer_client=console,
        mode=agent_mode,
        cycle_interval=0.5,
    )
    agent.configure_safety_limits(max_fader_step_db=3.0, max_fader_db=6.0, max_eq_step_db=2.0, max_comp_threshold_step_db=3.0)
    agent._max_actions_per_cycle = max(1, int(args.codex_orchestrator_max_actions))
    music_bed_lufs = _music_bed_lufs(plans)

    states = {}
    for ch, plan in plans.items():
        state = dict(plan.metrics)
        level_offset = plan.trim_db + plan.fader_db
        for key in ("peak_db", "true_peak_db", "rms_db", "lufs_momentary"):
            if key in state:
                state[key] = float(state[key]) + level_offset
        state.update({
            "channel_id": ch,
            "name": plan.name,
            "instrument": plan.instrument,
            "is_muted": False,
            "mix_lufs": music_bed_lufs,
            "vocal_target_delta_db": 7.0 if str(args.genre).strip().lower() == "rock" else 6.0,
        })
        states[ch] = state
    agent.update_channel_states_batch(states)
    codex_dry_run = _orchestrator_dry_run_enabled(args)

    orchestration_report = {
        "enabled": bool(args.codex_orchestrator) and not bool(args.source_rules_mert_only),
        "dry_run": codex_dry_run,
        "llm_enabled": llm is not None,
        "mode": agent.state.mode.value,
        "max_actions_per_cycle": agent._max_actions_per_cycle,
    }
    if args.source_rules_mert_only:
        orchestration_report.update({
            "enabled": False,
            "mode": "source_rules_mert_only",
            "proposed_actions": [],
            "proposed_count": 0,
            "applied_count": 0,
            "pending_before_approve": 0,
            "reason": "project_agent_and_rule_engine_disabled_for_this_render",
        })
    elif args.codex_orchestrator:
        try:
            proposed_actions = agent._prepare_actions(agent._decide({"channels": states}))
            orchestration_report.update({
                "proposed_actions": _agent_actions_to_dict(proposed_actions),
                "proposed_count": len(proposed_actions),
                "applied_count": 0,
            })
            if not codex_dry_run and proposed_actions:
                agent._queue_pending_actions(proposed_actions)
                orchestration_report["pending_before_approve"] = len(agent.state.pending_actions)
                approved = agent.approve_all_pending()
                orchestration_report["applied_count"] = approved
            else:
                orchestration_report["pending_before_approve"] = 0
            orchestration_report["mode"] = "orchestrated"
        except Exception as e:
            orchestration_report["error"] = str(e)
    else:
        actions = agent._prepare_actions(agent._decide({"channels": states}))
        asyncio.run(agent._act(actions))
        orchestration_report.update({
            "proposed_actions": _agent_actions_to_dict(actions),
            "proposed_count": len(actions),
            "applied_count": len(actions),
            "pending_before_approve": 0,
            "mode": "direct",
        })

    codex_corrections: dict[str, Any] = {
        "enabled": False,
        "actions": [],
        "requested": bool(args.codex_correction_pass),
        "notes": [
            "Legacy codex heuristic correction passes are disabled.",
            "Analyzer-only policy is active: additional corrections must come from measured AutoFOH detectors.",
        ] if args.codex_correction_pass else [],
    }
    codex_bleed_control = {
        "enabled": False,
        "requested": bool(args.codex_correction_pass),
        "notes": [
            "Legacy codex bleed-control heuristics are disabled.",
            "Use the AutoFOH analyzer pass and measured event-based dynamics instead.",
        ] if args.codex_correction_pass else [],
    }
    if args.source_rules_mert_only:
        disabled_reason = "disabled_by_source_rules_mert_only"
        event_based_dynamics = {"enabled": False, "reason": disabled_reason}
        autofoh_analyzer_pass = {"enabled": False, "reason": disabled_reason}
        bass_drum_push = {"enabled": False, "reason": disabled_reason}
        kick_presence_boost = {"enabled": False, "reason": disabled_reason}
        cymbal_focus_cleanup = {"enabled": False, "reason": disabled_reason}
        cross_adaptive_eq = {"enabled": False, "reason": disabled_reason}
        reference_mix_guidance = {"enabled": False, "reason": disabled_reason}
        genre_mix_profile = {"enabled": False, "reason": disabled_reason}
        kick_bass_hierarchy = {"enabled": False, "reason": disabled_reason}
        frequency_window_balance = {"enabled": False, "reason": disabled_reason}
        stem_mix_verification = {"enabled": False, "reason": disabled_reason}
        reference_vocal_fx_focus = {"enabled": False, "reason": disabled_reason}
    else:
        event_based_dynamics = apply_event_based_dynamics(plans)
        autofoh_analyzer_pass = (
            {"enabled": False, "notes": ["AutoFOH analyzer pass explicitly disabled by CLI flag."]}
            if args.no_autofoh_analyzer_pass
            else apply_autofoh_analyzer_pass(
                plans,
                target_len,
                sr,
                autofoh_config,
                max_rounds=args.autofoh_rounds,
                render_cache=render_cache,
                analysis_preview_sec=analysis_preview_sec,
            )
        )
        bass_drum_push = apply_bass_drum_push(plans, args.bass_drum_boost_db)
        kick_presence_boost = apply_kick_presence_boost(plans, args.kick_presence_boost_db)
        cymbal_focus_cleanup = apply_cymbal_cleanup_for_kick_focus(plans, args.kick_focus_cymbal_cut_db)
        cross_adaptive_eq = apply_cross_adaptive_eq(
            plans,
            target_len,
            sr,
            render_cache=render_cache,
            analysis_preview_sec=analysis_preview_sec,
        )
        reference_mix_guidance = apply_reference_mix_guidance(plans, sr, reference_context)
        genre_mix_profile = apply_genre_mix_profile(plans, args.genre)
        kick_bass_hierarchy = apply_kick_bass_hierarchy(
            plans,
            target_len,
            sr,
            desired_kick_advantage_db=desired_kick_advantage_from_reference(
                reference_context,
                genre=args.genre,
            ),
            render_cache=render_cache,
            analysis_preview_sec=analysis_preview_sec,
        )
        frequency_window_balance = apply_frequency_window_balance(
            plans,
            target_len,
            sr,
            render_cache=render_cache,
        )
        stem_mix_verification = apply_stem_mix_verification(
            plans,
            target_len,
            sr,
            genre=args.genre,
            reference_context=reference_context,
            render_cache=render_cache,
            analysis_preview_sec=analysis_preview_sec,
        )
        reference_vocal_fx_focus = (
            apply_reference_vocal_fx_focus(
                plans,
                target_len,
                sr,
                reference_context,
                genre=args.genre,
                render_cache=render_cache,
                analysis_preview_sec=analysis_preview_sec,
            )
            if args.reference_vocal_fx_focus
            else {"enabled": False, "reason": "disabled_by_flag"}
        )
    vocal_bed_balance = {
        "enabled": False,
        "notes": [
            "Static vocal bed attenuation is disabled.",
            "Vocal space must come from priority EQ and measured analyzer corrections.",
        ],
    }
    live_channel_headroom = (
        apply_live_channel_peak_headroom(
            plans,
            target_len,
            sr,
            args.live_channel_peak_ceiling_db,
            render_cache=render_cache,
        )
        if args.no_final_limiter
        else {"enabled": False}
    )

    rendered_channels: dict[int, np.ndarray] = {}
    source_candidate_logs = 0
    for ch, plan in plans.items():
        if plan.muted:
            continue
        source_candidate_logs += trace_channel_source_candidates(
            source_layer,
            session_id=source_session_id,
            channel=ch,
            plan=plan,
            target_len=target_len,
            sr=sr,
        )
        rendered = render_channel_cached(ch, plan, target_len, sr, render_cache)
        rendered_channels[ch] = rendered

    layer_group_base_target = (
        float(source_rules_mert_only_report.get("base_target_rms_db", -25.0))
        if args.source_rules_mert_only
        else _infer_layer_group_base_target(plans)
    )
    layer_group_mix = apply_layer_group_mix_corrections(
        rendered_channels,
        plans,
        sr,
        base_target_rms_db=layer_group_base_target,
        band_medians=(
            source_rules_mert_only_report.get("band_medians") or None
            if args.source_rules_mert_only
            else None
        ),
        source_layer=source_layer,
        source_session_id=source_session_id,
    )
    layer_group_mix["base_target_rms_db"] = round(float(layer_group_base_target), 3)
    if args.source_rules_mert_only:
        source_rules_mert_only_report["layer_group_mix"] = layer_group_mix

    channel_reports = []
    for ch, plan in plans.items():
        if plan.muted:
            continue
        channel_reports.append({
            "channel": ch,
            "file": plan.path.name,
            "instrument": plan.instrument,
            "trim_db": round(plan.trim_db, 2),
            "trim_analysis": plan.trim_analysis,
            "agent_fader_db": round(plan.fader_db, 2),
            "pan": plan.pan,
            "hpf": plan.hpf,
            "lpf": plan.lpf,
            "phase_invert": plan.phase_invert,
            "delay_ms": round(plan.delay_ms, 3),
            "input_fade_ms": round(plan.input_fade_ms, 2),
            "compressor": {
                "threshold_db": round(plan.comp_threshold_db, 2),
                "ratio": round(plan.comp_ratio, 2),
                "attack_ms": round(plan.comp_attack_ms, 2),
                "release_ms": round(plan.comp_release_ms, 2),
            },
            "event_expander": plan.expander_report,
            "phase_notes": plan.phase_notes,
            "pan_notes": plan.pan_notes,
            "autofoh_actions": plan.autofoh_actions,
            "cross_adaptive_eq": plan.cross_adaptive_eq,
            "peak_db": round(plan.metrics["peak_db"], 2),
            "rms_db": round(plan.metrics["rms_db"], 2),
            "analysis_mode": plan.metrics.get("analysis_mode"),
            "analysis_active_sec": plan.metrics.get("analysis_active_sec"),
            "analysis_active_ratio": plan.metrics.get("analysis_active_ratio"),
            "analysis_threshold_db": plan.metrics.get("analysis_threshold_db"),
        })

    dynamic_vocal_priority = {
        "enabled": False,
        "requested_disable_flag": bool(args.no_drum_vocal_duck),
        "notes": [
            "Vocal ducking is disabled.",
            "Vocal space must be created by priority EQ and measured analyzer corrections.",
        ],
    }
    reference_dynamics_ride = (
        {"enabled": False, "reason": "disabled_by_source_rules_mert_only"}
        if args.source_rules_mert_only
        else (
            apply_reference_dynamics_ride(rendered_channels, plans, sr, reference_context)
            if args.reference_dynamics_ride
            else {"enabled": False, "reason": "disabled_by_flag"}
        )
    )
    source_rules_only_fx_plan = (
        build_source_rules_only_fx_plan(plans, tempo_bpm=args.tempo_bpm)
        if args.source_rules_mert_only
        else None
    )
    fx_returns, fx_report = (
        ({}, {"enabled": False})
        if args.disable_fx
        else apply_offline_fx_plan(
            rendered_channels,
            plans,
            sr,
            tempo_bpm=args.tempo_bpm,
            reference_context=reference_context,
            source_layer=source_layer,
            source_session_id=source_session_id,
            fx_plan_override=source_rules_only_fx_plan,
        )
    )
    large_system_polish = (
        {"enabled": False, "reason": "disabled_by_source_rules_mert_only"}
        if args.source_rules_mert_only
        else (
            apply_large_system_translation_polish(
                rendered_channels,
                fx_returns,
                plans,
                sr,
                reference_context=reference_context,
                genre=args.genre,
            )
            if args.large_system_polish
            else {"enabled": False, "reason": "disabled_by_flag"}
        )
    )
    mix = np.zeros((target_len, 2), dtype=np.float32)
    for rendered in rendered_channels.values():
        mix += rendered
    for rendered in fx_returns.values():
        mix += rendered
    perceptual_before_master = mix.copy()

    final_limiter = (not args.no_final_limiter) or args.soft_master
    mix, master_report = master_process(
        mix,
        sr,
        target_lufs=args.master_target_lufs,
        final_limiter=final_limiter,
        live_peak_ceiling_db=args.live_peak_ceiling_db,
        reference_context=reference_context,
        allow_reference_mastering=not args.no_reference_mastering,
        ceiling_dbfs=args.master_ceiling_dbfs,
        apply_bus_processing=not args.source_rules_mert_only,
    )

    output = Path(args.output)
    if output.suffix.lower() == ".wav":
        tmp_wav = output
        sf.write(tmp_wav, mix, sr, subtype="PCM_24")
    else:
        tmp_wav = output.with_suffix(".wav")
        sf.write(tmp_wav, mix, sr, subtype="PCM_24")
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_wav),
            "-codec:a", "libmp3lame",
            "-b:a", "320k",
            str(output),
        ]
        subprocess.run(cmd, check=True)

    meter = pyln.Meter(sr)
    final_lufs = None
    try:
        final_lufs = float(meter.integrated_loudness(mix))
    except Exception:
        pass
    perceptual_mix_bus = _record_offline_perceptual_mix_bus(
        perceptual_evaluator,
        before_audio=perceptual_before_master,
        after_audio=mix,
        sr=sr,
        context={
            "channel": "mix_bus",
            "instrument": "full_mix",
            "sample_rate": sr,
            "session_id": source_session_id,
            "action": {
                "action_type": "offline_mert_mix_bus_shadow",
                "master_target_lufs": round(float(args.master_target_lufs), 3),
                "master_ceiling_dbfs": round(float(args.master_ceiling_dbfs), 3),
                "final_limiter": bool(final_limiter),
                "large_system_polish": bool(args.large_system_polish),
                "source_knowledge_enabled": bool(source_layer and source_layer.enabled),
                "source_rules_mert_only": bool(args.source_rules_mert_only),
            },
            "engineering_score": 0.0,
            "safety_score": 1.0,
        },
    )
    perceptual_report["mix_bus"] = perceptual_mix_bus
    if source_layer is not None:
        source_layer.stop()
        source_knowledge_report["channel_candidate_logs_requested"] = int(source_candidate_logs)
        source_knowledge_report["logger_stats"] = source_layer.logger.stats.__dict__.copy()
    report = {
        "input_dir": str(input_dir),
        "output": str(output),
        "reference": str(reference_context.path) if reference_context is not None else "",
        "reference_sources": [str(path) for path in reference_context.source_paths] if reference_context is not None else [],
        "genre": str(args.genre or ""),
        "genre_reference_seed": genre_reference_seed,
        "sample_rate": sr,
        "performance": {
            "autofoh_rounds": max(1, int(args.autofoh_rounds)),
            "analysis_preview_sec": round(float(analysis_preview_sec), 2),
            "render_cache": render_cache.summary(),
            "fast_compressor": os.environ.get("AUTO_MIXER_SLOW_COMPRESSOR") != "1",
        },
        "codex_orchestrator": orchestration_report,
        "source_rules_mert_only": source_rules_mert_only_report,
        "layer_group_mix": layer_group_mix,
        "source_knowledge": source_knowledge_report,
        "perceptual": perceptual_report,
        "duration_sec": round(target_len / sr, 3),
        "final_peak_dbfs": round(amp_to_db(float(np.max(np.abs(mix)))), 2),
        "final_lufs": round(final_lufs, 2) if final_lufs is not None and np.isfinite(final_lufs) else None,
        "music_bed_lufs": round(_music_bed_lufs(plans), 2),
        "vocal_bed_balance": vocal_bed_balance,
        "bass_drum_boost": bass_drum_push,
        "cross_adaptive_eq": cross_adaptive_eq,
        "reference_mix_guidance": reference_mix_guidance,
        "genre_mix_profile": genre_mix_profile,
        "kick_bass_hierarchy": kick_bass_hierarchy,
        "frequency_window_balance": frequency_window_balance,
        "stem_mix_verification": stem_mix_verification,
        "reference_vocal_fx_focus": reference_vocal_fx_focus,
        "reference_dynamics_ride": reference_dynamics_ride,
        "large_system_polish": large_system_polish,
        "dynamic_vocal_priority": dynamic_vocal_priority,
        "kick_presence_boost": kick_presence_boost,
        "kick_focus_cymbal_cut": cymbal_focus_cleanup,
        "soft_master": args.soft_master,
        "master_target_lufs": round(args.master_target_lufs, 2),
        "master_ceiling_dbfs": round(args.master_ceiling_dbfs, 2),
        "channel_input_fade_ms": round(args.channel_input_fade_ms, 2),
        "autofoh_analyzer_pass": autofoh_analyzer_pass,
        "codex_corrections": codex_corrections,
        "codex_bleed_control": codex_bleed_control,
        "event_based_dynamics": event_based_dynamics,
        "fx": fx_report,
        "master_processing": master_report,
        "live_channel_headroom": live_channel_headroom,
        "phase_alignment": phase_report,
        "drum_pan_rule": drum_pan_rule,
        "agent_actions": agent.get_action_history(100),
        "agent_audit": agent.get_action_audit_log(100),
        "virtual_console_calls": console.calls,
        "channels": channel_reports,
    }
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
