"""
Auto Soundcheck Engine — headless orchestrator for automatic mixing.

Connects to mixer (dLive or WING), detects audio device (SoundGrid/Dante),
reads channel names, recognizes instruments, waits for audio signals,
and automatically applies gain staging, EQ, compressor, and fader corrections
without user confirmation.

This is the main entry point for fully automatic operation.
"""

import asyncio
import json
import logging
import time
import threading
from pathlib import Path
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from audio_capture import (
    AudioCapture, AudioSourceType, AudioDeviceType,
    detect_audio_device, find_device_by_name, list_audio_devices,
)
from autofoh_analysis import build_stem_contribution_matrix, extract_analysis_features
from autofoh_detectors import (
    HarshnessExcessDetector,
    LeadMaskingAnalyzer,
    LowEndAnalyzer,
    MudExcessDetector,
    SibilanceExcessDetector,
    aggregate_stem_features,
)
from autofoh_evaluation import (
    ActionEvaluationOutcome,
    AutoFOHEvaluationPolicy,
    PendingActionEvaluation,
    build_rollback_action,
    evaluate_pending_action,
    infer_evaluation_band,
)
from autofoh_logging import (
    AutoFOHSessionReport,
    AutoFOHStructuredLogger,
    build_session_report_from_jsonl,
    render_session_report_summary,
    write_session_report,
)
from autofoh_models import RuntimeState, TargetCorridor
from autofoh_profiles import (
    AutoFOHSoundcheckProfile,
    AutoFOHSoundcheckProfileStore,
    PhaseLearningSnapshot,
    build_phase_learning_snapshot,
    build_soundcheck_profile,
)
from autofoh_safety import (
    AutoFOHSafetyConfig,
    AutoFOHSafetyController,
    BusCompressorAdjust,
    BusCompressorMakeupAdjust,
    BusEQMove,
    BusFaderMove,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    CompressorAdjust,
    CompressorMakeupAdjust,
    DCAFaderMove,
    DelayAdjust,
    EmergencyFeedbackNotch,
    GateAdjust,
    HighPassAdjust,
    MasterFaderMove,
    PanAdjust,
    PolarityAdjust,
    SafetyDecision,
    SendLevelAdjust,
)
from live_shared_mix import (
    LiveSharedMixChannel,
    LiveSharedMixConfig,
    build_live_shared_mix_plan,
    normalize_live_role,
)
from channel_recognizer import (
    classification_from_legacy_preset,
    recognize_instrument, recognize_instrument_spectral_fallback,
    scan_and_recognize, AVAILABLE_PRESETS,
)
from config_manager import ConfigManager
from feedback_detector import FeedbackDetector, FeedbackDetectorConfig, FeedbackEvent
from heuristics.spectral_ceiling_eq import (
    SpectralCeilingEQAnalyzer,
    SpectralCeilingEQConfig,
    format_spectral_ceiling_log,
    merge_spectral_proposal_into_eq_bands,
)
from mixer_discovery import (
    discover_mixer_auto, discover_mixers, DiscoveredMixer,
)
from audio_device_scanner import (
    scan_audio_devices, select_best_device, detect_and_report,
    AudioDevice, AudioProtocol,
)
from signal_metrics import (
    SignalAnalyzer, ChannelMetrics, compare_channels,
    InterChannelMetrics, LevelMetrics, DynamicsMetrics, SpectralMetrics,
)
from observation_mixer import ObservationMixerClient

try:
    from automixer.config import load_decision_engine_v2_config
    from automixer.decision import DecisionEngine
    from automixer.executor import ActionPlanExecutor
    from automixer.knowledge import MixingKnowledgeBase
    from automixer.logs import HumanDecisionLogger
    from automixer.safety import SafetyGate
except Exception:
    load_decision_engine_v2_config = None
    DecisionEngine = None
    ActionPlanExecutor = None
    MixingKnowledgeBase = None
    HumanDecisionLogger = None
    SafetyGate = None

try:
    from perceptual import PerceptualEvaluator
except Exception:
    PerceptualEvaluator = None

try:
    from evaluation import MuQEvalService
except Exception:
    MuQEvalService = None

logger = logging.getLogger(__name__)


class EngineState(Enum):
    IDLE = "idle"
    DISCOVERING = "discovering"
    CONNECTING = "connecting"
    SCANNING_AUDIO = "scanning_audio"
    SCANNING_CHANNELS = "scanning_channels"
    READING_STATE = "reading_state"
    RESETTING = "resetting"
    WAITING_FOR_SIGNAL = "waiting_for_signal"
    ANALYZING = "analyzing"
    APPLYING = "applying"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class ChannelSnapshot:
    """Snapshot of a channel's settings before automatic correction."""
    channel: int
    fader_db: float = -100.0
    muted: bool = False
    eq_bands: Optional[List[Tuple[float, float, float]]] = None
    hpf_freq: float = 20.0
    hpf_enabled: bool = False
    gain_db: float = 0.0
    had_processing: bool = False
    raw_settings: Optional[Dict] = None


@dataclass
class BusSnapshot:
    """Snapshot of a bus that receives corrected channels."""
    bus: int
    name: str = ""
    fader_db: float = -100.0
    muted: bool = False
    eq_bands: Optional[List[Tuple[float, float, float]]] = None
    compressor_enabled: bool = False
    dca_assignments: List[int] = field(default_factory=list)
    source_channels: List[int] = field(default_factory=list)
    raw_settings: Optional[Dict] = None


@dataclass
class DCASnapshot:
    """Snapshot of a DCA controlling corrected channels or their buses."""
    dca: int
    name: str = ""
    fader_db: float = -100.0
    muted: bool = False
    source_channels: List[int] = field(default_factory=list)
    source_buses: List[int] = field(default_factory=list)
    raw_settings: Optional[Dict] = None


@dataclass
class ChannelInfo:
    """Per-channel information gathered during soundcheck."""
    channel: int
    name: str = ""
    preset: Optional[str] = None
    source_role: str = "unknown"
    stem_roles: List[str] = field(default_factory=list)
    allowed_controls: List[str] = field(default_factory=list)
    priority: float = 0.0
    classification_confidence: float = 0.0
    classification_match_type: str = "unknown"
    auto_corrections_enabled: bool = False
    recognized: bool = False
    has_signal: bool = False
    peak_db: float = -100.0
    rms_db: float = -100.0
    lufs: float = -100.0
    gain_correction_db: float = 0.0
    eq_applied: bool = False
    compressor_applied: bool = False
    fader_db: float = -100.0
    spectral_centroid: float = 0.0
    was_reset: bool = False
    original_snapshot: Optional[ChannelSnapshot] = None
    metrics: Optional[ChannelMetrics] = None
    analyzer: Optional[Any] = None


# Instrument-specific EQ presets: (freq_hz, gain_db, q)
# 4-band PEQ per instrument type
INSTRUMENT_EQ_PRESETS: Dict[str, List[Tuple[float, float, float]]] = {
    "kick": [
        (60.0, 3.0, 1.0),      # Sub thump
        (200.0, -3.0, 2.0),    # Cut mud
        (3500.0, 2.0, 2.0),    # Attack click
        (8000.0, -2.0, 1.5),   # Reduce bleed
    ],
    "snare": [
        (120.0, 1.5, 1.5),     # Body
        (400.0, -2.0, 2.0),    # Cut boxiness
        (2500.0, 2.0, 2.5),    # Crack
        (8000.0, 1.0, 1.5),    # Sizzle
    ],
    "tom": [
        (80.0, 2.0, 1.2),      # Low body
        (300.0, -2.5, 2.0),    # Reduce boxiness
        (3000.0, 1.5, 2.0),    # Attack
        (7000.0, -1.5, 1.5),   # Reduce bleed
    ],
    "hihat": [
        (200.0, -4.0, 1.0),    # HPF slope assist
        (1000.0, -1.0, 1.5),   # Cut honk
        (6000.0, 1.5, 2.0),    # Presence
        (12000.0, 1.0, 1.0),   # Air
    ],
    "ride": [
        (200.0, -3.0, 1.0),    # HPF slope assist
        (800.0, -1.0, 2.0),    # Cut mud
        (3000.0, 1.0, 2.0),    # Definition
        (10000.0, 1.5, 1.5),   # Shimmer
    ],
    "cymbals": [
        (200.0, -3.0, 1.0),    # Cut lows
        (1000.0, -1.5, 2.0),   # Cut mid
        (5000.0, 1.0, 2.0),    # Presence
        (12000.0, 1.0, 1.0),   # Air
    ],
    "overheads": [
        (100.0, -2.0, 1.0),    # Trim lows
        (500.0, -1.0, 1.5),    # Cut mud
        (4000.0, 1.5, 2.0),    # Presence
        (12000.0, 1.0, 1.0),   # Air
    ],
    "room": [
        (80.0, -3.0, 0.8),     # Cut rumble
        (400.0, -1.0, 1.5),    # Cut box
        (2000.0, 1.0, 2.0),    # Clarity
        (10000.0, 0.5, 1.0),   # Air
    ],
    "bass": [
        (60.0, 2.0, 1.2),      # Sub
        (250.0, -2.0, 2.0),    # Cut mud
        (800.0, 1.0, 2.0),     # Growl/definition
        (3000.0, 1.5, 2.5),    # Pick attack
    ],
    "electricGuitar": [
        (100.0, -2.0, 1.0),    # Cut sub lows
        (500.0, -1.5, 2.0),    # Reduce mud
        (3000.0, 2.0, 2.0),    # Presence
        (6000.0, 1.0, 1.5),    # Bite
    ],
    "acousticGuitar": [
        (80.0, -3.0, 1.0),     # Cut rumble
        (250.0, -1.5, 2.0),    # Reduce body boom
        (3000.0, 2.0, 2.0),    # String definition
        (8000.0, 1.0, 1.5),    # Air
    ],
    "accordion": [
        (100.0, -2.0, 1.0),    # Cut sub
        (500.0, -1.0, 2.0),    # Reduce box
        (2500.0, 1.5, 2.0),    # Reed clarity
        (6000.0, 1.0, 1.5),    # Presence
    ],
    "synth": [
        (40.0, -1.0, 1.0),     # Trim sub
        (300.0, -1.0, 2.0),    # Reduce mud
        (2000.0, 1.0, 2.0),    # Clarity
        (8000.0, 1.5, 1.5),    # Brightness
    ],
    "playback": [
        (60.0, 0.0, 1.0),      # Flat low
        (300.0, -1.0, 1.5),    # Slight mud cut
        (3000.0, 0.5, 2.0),    # Slight presence
        (10000.0, 0.5, 1.0),   # Air
    ],
    "leadVocal": [
        (80.0, -4.0, 0.8),     # HPF assist
        (300.0, -2.0, 2.0),    # Reduce mud
        (3000.0, 2.5, 2.5),    # Presence
        (8000.0, 1.5, 1.5),    # Air
    ],
    "backVocal": [
        (100.0, -3.0, 0.8),    # HPF assist
        (400.0, -2.0, 2.0),    # Reduce mud
        (2500.0, 1.5, 2.0),    # Clarity
        (6000.0, 1.0, 1.5),    # Air
    ],
    "custom": [
        (60.0, 0.0, 1.0),
        (300.0, -1.0, 1.5),
        (3000.0, 0.5, 2.0),
        (10000.0, 0.5, 1.0),
    ],
}

# HPF frequencies per instrument type
INSTRUMENT_HPF: Dict[str, float] = {
    "kick": 30.0,
    "snare": 80.0,
    "tom": 60.0,
    "hihat": 300.0,
    "ride": 250.0,
    "cymbals": 300.0,
    "overheads": 120.0,
    "room": 80.0,
    "bass": 30.0,
    "electricGuitar": 80.0,
    "acousticGuitar": 80.0,
    "accordion": 80.0,
    "synth": 30.0,
    "playback": 30.0,
    "leadVocal": 80.0,
    "backVocal": 100.0,
    "custom": 80.0,
}

# Target LUFS per instrument category
INSTRUMENT_TARGET_LUFS: Dict[str, float] = {
    "kick": -25.0,
    "snare": -25.0,
    "tom": -27.0,
    "hihat": -35.0,
    "ride": -35.0,
    "cymbals": -35.0,
    "overheads": -30.0,
    "room": -35.0,
    "bass": -23.0,
    "electricGuitar": -23.0,
    "acousticGuitar": -25.0,
    "accordion": -23.0,
    "synth": -22.0,
    "playback": -23.0,
    "leadVocal": -20.0,
    "backVocal": -23.0,
    "custom": -23.0,
}

# Compressor presets per instrument: (threshold_db, ratio, attack_ms, release_ms)
INSTRUMENT_COMPRESSOR: Dict[str, Tuple[float, float, float, float]] = {
    "kick": (-20.0, 4.0, 10.0, 80.0),
    "snare": (-18.0, 3.5, 5.0, 100.0),
    "tom": (-18.0, 3.5, 8.0, 100.0),
    "hihat": (-25.0, 2.5, 5.0, 60.0),
    "ride": (-25.0, 2.0, 10.0, 100.0),
    "cymbals": (-25.0, 2.0, 10.0, 100.0),
    "overheads": (-20.0, 2.0, 15.0, 150.0),
    "room": (-20.0, 3.0, 10.0, 200.0),
    "bass": (-15.0, 4.0, 10.0, 100.0),
    "electricGuitar": (-18.0, 3.0, 10.0, 100.0),
    "acousticGuitar": (-18.0, 3.0, 15.0, 150.0),
    "accordion": (-18.0, 2.5, 15.0, 150.0),
    "synth": (-18.0, 2.5, 10.0, 100.0),
    "playback": (-20.0, 2.0, 10.0, 100.0),
    "leadVocal": (-15.0, 3.0, 5.0, 80.0),
    "backVocal": (-18.0, 3.0, 5.0, 80.0),
    "custom": (-20.0, 2.5, 10.0, 100.0),
}

# Close mics where a gate/expander is part of hearing the corrected result.
# Tuple: threshold_db, range_db, attack_ms, hold_ms, release_ms.
INSTRUMENT_GATE_PRESETS: Dict[str, Tuple[float, float, float, float, float]] = {
    "kick": (-36.0, 32.0, 1.0, 70.0, 140.0),
    "snare": (-42.0, 26.0, 1.5, 80.0, 180.0),
    "tom": (-44.0, 34.0, 2.0, 120.0, 260.0),
}

# Default pan positions per instrument (-100=L, 0=C, +100=R)
INSTRUMENT_PAN: Dict[str, float] = {
    "kick": 0.0,
    "snare": 0.0,
    "tom": -15.0,       # Slightly left (Tom 1 default; engine adjusts per Tom number)
    "hihat": 30.0,      # Slightly right (audience perspective)
    "ride": -30.0,      # Slightly left
    "cymbals": 0.0,
    "overheads": 0.0,   # Handled as stereo pair: L=-60, R=+60
    "room": 0.0,
    "bass": 0.0,
    "electricGuitar": -25.0,
    "acousticGuitar": 25.0,
    "accordion": 15.0,
    "synth": 0.0,       # Stereo pair: L=-40, R=+40
    "playback": 0.0,
    "leadVocal": 0.0,
    "backVocal": 0.0,   # Spread: BVox1=-30, BVox2=+30
    "custom": 0.0,
}

# Target input gain (preamp trim) in dB per instrument
INSTRUMENT_TARGET_INPUT_DB: Dict[str, float] = {
    "kick": -10.0,
    "snare": -10.0,
    "tom": -10.0,
    "hihat": -15.0,
    "ride": -15.0,
    "cymbals": -15.0,
    "overheads": -15.0,
    "room": -10.0,
    "bass": -10.0,
    "electricGuitar": -10.0,
    "acousticGuitar": -10.0,
    "accordion": -10.0,
    "synth": -15.0,
    "playback": -15.0,
    "leadVocal": -8.0,
    "backVocal": -10.0,
    "custom": -12.0,
}

# FX send presets: reverb bus, delay bus (send_bus_number, level_db)
# Assumes: Bus 1 = Reverb (plate/hall), Bus 2 = Delay
INSTRUMENT_FX_SENDS: Dict[str, List[Tuple[int, float]]] = {
    "kick": [],
    "snare": [(1, -20.0)],            # Light reverb
    "tom": [(1, -22.0)],
    "hihat": [],
    "ride": [],
    "cymbals": [],
    "overheads": [(1, -25.0)],
    "room": [],
    "bass": [],
    "electricGuitar": [(1, -18.0), (2, -25.0)],   # Reverb + delay
    "acousticGuitar": [(1, -15.0)],
    "accordion": [(1, -18.0)],
    "synth": [(1, -20.0), (2, -22.0)],
    "playback": [],
    "leadVocal": [(1, -12.0), (2, -18.0)],         # More reverb + delay
    "backVocal": [(1, -15.0), (2, -20.0)],
    "custom": [],
}

# Channels that typically come in correlated pairs (for phase check)
STEREO_PAIR_PRESETS = ["overheads", "synth", "playback"]

SIGNAL_THRESHOLD_DB = -50.0
SIGNAL_ANALYSIS_SECONDS = 3.0
GAIN_CORRECTION_MAX_DB = 12.0


class AutoSoundcheckEngine:
    """
    Fully automatic soundcheck engine.

    Workflow:
    0. Auto-discover mixer on the network (WING broadcast / dLive TCP scan)
    1. Connect to mixer (dLive / WING)
    2. Detect audio device (SoundGrid / Dante)
    3. Start audio capture
    4. Read channel names from mixer → recognize instruments
    5. Wait for audio signal on each channel
    6. For channels with signal:
       a. Analyze spectrum → spectral instrument classification fallback
       b. Compute gain correction (target LUFS per instrument)
       c. Apply HPF, EQ preset, compressor preset
       d. Set fader to initial position
    7. Enter running mode: continuous feedback detection + level monitoring
    """

    def __init__(
        self,
        mixer_type: Optional[str] = None,
        mixer_ip: Optional[str] = None,
        mixer_port: Optional[int] = None,
        mixer_tls: bool = False,
        midi_base_channel: int = 0,
        audio_device_name: Optional[str] = None,
        num_channels: int = 48,
        sample_rate: int = 48000,
        block_size: int = 1024,
        buffer_seconds: float = 5.0,
        auto_apply: bool = True,
        observe_only: bool = False,
        selected_channels: Optional[List[int]] = None,
        auto_discover: bool = True,
        scan_subnet: bool = True,
        on_state_change: Optional[Callable] = None,
        on_channel_update: Optional[Callable] = None,
        on_observation: Optional[Callable] = None,
        config_path: Optional[str] = None,
        use_decision_engine_v2: bool = False,
        decision_engine_dry_run: bool = False,
    ):
        self.mixer_type = mixer_type
        self.mixer_ip = mixer_ip
        self.mixer_port = mixer_port
        self.mixer_tls = mixer_tls
        self.midi_base_channel = midi_base_channel
        self.audio_device_name = audio_device_name
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.buffer_seconds = buffer_seconds
        self.auto_apply = auto_apply
        self.observe_only = observe_only
        self.auto_discover = auto_discover
        self.scan_subnet = scan_subnet
        self.selected_channels = sorted({
            int(ch) for ch in (selected_channels or [])
            if int(ch) > 0
        })
        self.use_decision_engine_v2 = bool(use_decision_engine_v2)
        self.decision_engine_v2_dry_run_requested = bool(decision_engine_dry_run)

        default_config_path = Path(config_path) if config_path else (
            Path(__file__).resolve().parents[1] / "config" / "automixer.yaml"
        )
        resolved_config_path = str(default_config_path) if default_config_path.exists() else config_path
        self.config_manager = ConfigManager(config_path=resolved_config_path)
        self.decision_engine_v2_config = (
            load_decision_engine_v2_config(resolved_config_path)
            if load_decision_engine_v2_config is not None
            else None
        )
        if self.decision_engine_v2_config is not None:
            self.use_decision_engine_v2 = bool(
                self.use_decision_engine_v2 or self.decision_engine_v2_config.enabled
            )
            self.decision_engine_v2_dry_run = bool(
                self.decision_engine_v2_dry_run_requested
                or self.decision_engine_v2_config.dry_run
            )
        else:
            self.decision_engine_v2_dry_run = self.decision_engine_v2_dry_run_requested
        autofoh_config = self.config_manager.get_section("autofoh")
        self.spectral_ceiling_eq_config = SpectralCeilingEQConfig.from_mapping(
            self.config_manager.get_section("spectral_ceiling_eq")
        )
        try:
            self.spectral_ceiling_eq_analyzer: Optional[SpectralCeilingEQAnalyzer] = (
                SpectralCeilingEQAnalyzer(self.spectral_ceiling_eq_config)
            )
        except Exception as exc:
            logger.warning("Spectral ceiling EQ disabled; profile load failed: %s", exc)
            self.spectral_ceiling_eq_config.enabled = False
            self.spectral_ceiling_eq_analyzer = None
        self.classifier_config = autofoh_config.get("classifier", {})
        autofoh_safety = autofoh_config.get("safety", {})
        autofoh_evaluation = autofoh_config.get("evaluation", {})
        autofoh_logging = autofoh_config.get("logging", {})
        autofoh_soundcheck_profile = autofoh_config.get("soundcheck_profile", {})
        self.shared_chat_mix_config = LiveSharedMixConfig.from_mapping(
            autofoh_config.get("shared_chat_mix", {})
        )
        perceptual_config = self.config_manager.get_section("perceptual")
        muq_eval_config = self._load_muq_eval_config()
        self.autofoh_analysis_config = autofoh_config.get("analysis", {})
        detector_config = autofoh_config.get("detectors", {})
        self.monitor_cycle_interval_sec = float(
            detector_config.get("monitor_cycle_interval_sec", 1.0)
        )
        self.feedback_event_min_interval_sec = float(
            detector_config.get("feedback_event_min_interval_sec", 8.0)
        )
        self.feedback_frequency_bucket_hz = float(
            detector_config.get("feedback_frequency_bucket_hz", 100.0)
        )
        self.feedback_channel_min_interval_sec = float(
            detector_config.get("feedback_channel_min_interval_sec", 8.0)
        )
        self.feedback_global_min_interval_sec = float(
            detector_config.get("feedback_global_min_interval_sec", 0.75)
        )
        self.feedback_max_actions_per_channel_per_run = int(
            detector_config.get("feedback_max_actions_per_channel_per_run", 2)
        )
        self.feedback_max_actions_per_run = int(
            detector_config.get("feedback_max_actions_per_run", 16)
        )
        self.feedback_detector_persistence_frames = int(
            detector_config.get("feedback_persistence_frames", 12)
        )
        self.feedback_detector_min_confidence = float(
            detector_config.get("feedback_min_confidence", 0.75)
        )
        self.feedback_detector_peak_height_db = float(
            detector_config.get("feedback_peak_height_db", -12.0)
        )
        self.feedback_detector_peak_prominence_db = float(
            detector_config.get("feedback_peak_prominence_db", 10.0)
        )
        self.minimum_auto_apply_classification_confidence = float(
            autofoh_safety.get("minimum_auto_apply_classification_confidence", 0.75)
        )
        self.new_or_unknown_channel_auto_corrections_enabled = bool(
            autofoh_safety.get("new_or_unknown_channel_auto_corrections_enabled", False)
        )
        self.preserve_existing_processing = bool(
            autofoh_safety.get("preserve_existing_processing", True)
        )
        self.allow_destructive_reset = bool(
            autofoh_safety.get("allow_destructive_reset", False)
        )
        self.allow_input_trim_boost = bool(
            autofoh_safety.get("allow_input_trim_boost", False)
        )
        self.auto_gate_close_mics = bool(
            autofoh_safety.get("auto_gate_close_mics", True)
        )
        self.auto_fx_sends_enabled = bool(
            autofoh_safety.get("auto_fx_sends_enabled", False)
        )
        self.close_mic_fx_sends_enabled = bool(
            autofoh_safety.get("close_mic_fx_sends_enabled", False)
        )
        self.monitor_bus_ids = {
            int(bus) for bus in autofoh_safety.get("monitor_bus_ids", [13, 14, 15, 16])
            if int(bus) > 0
        }
        self.effect_bus_ids = {
            int(bus) for bus in autofoh_safety.get("effect_bus_ids", [9, 10])
            if int(bus) > 0
        }
        self.group_bus_correction_enabled = bool(
            autofoh_safety.get("group_bus_correction_enabled", True)
        )
        self.master_reference_channels = {
            int(ch) for ch in autofoh_safety.get("master_reference_channels", [])
            if int(ch) > 0
        }
        self.compressor_gr_max_db = float(
            autofoh_safety.get("compressor_gr_max_db", 8.0)
        )
        self.compressor_gr_target_db = float(
            autofoh_safety.get("compressor_gr_target_db", 4.0)
        )
        self.compressor_makeup_compensation_ratio = float(
            autofoh_safety.get("compressor_makeup_compensation_ratio", 0.7)
        )
        self.compressor_makeup_max_step_db = float(
            autofoh_safety.get("compressor_makeup_max_step_db", 2.0)
        )
        self.compressor_makeup_max_db = float(
            autofoh_safety.get("compressor_makeup_max_db", 9.0)
        )
        self.compressor_makeup_true_peak_ceiling_db = float(
            autofoh_safety.get("compressor_makeup_true_peak_ceiling_db", -3.0)
        )
        self.action_safety_config = AutoFOHSafetyConfig.from_config(
            autofoh_safety.get("action_limits", {})
        )
        self.evaluation_policy = AutoFOHEvaluationPolicy.from_config(autofoh_evaluation)
        self.autofoh_logging_enabled = bool(autofoh_logging.get("enabled", False))
        self.autofoh_log_path = str(autofoh_logging.get("path", "") or "")
        self.autofoh_log_queue_maxsize = int(autofoh_logging.get("queue_maxsize", 1024))
        self.autofoh_write_session_report_on_stop = bool(
            autofoh_logging.get("write_session_report_on_stop", True)
        )
        self.autofoh_report_path = str(autofoh_logging.get("report_path", "") or "")
        self.soundcheck_profile_enabled = bool(
            autofoh_soundcheck_profile.get("enabled", True)
        )
        self.soundcheck_profile_auto_save = bool(
            autofoh_soundcheck_profile.get("auto_save_after_analysis", True)
        )
        self.soundcheck_profile_auto_load = bool(
            autofoh_soundcheck_profile.get("auto_load_on_start", True)
        )
        self.soundcheck_profile_capture_multiphase_learning = bool(
            autofoh_soundcheck_profile.get("capture_multiphase_learning", True)
        )
        self.soundcheck_profile_silence_capture_duration_sec = float(
            autofoh_soundcheck_profile.get("silence_capture_duration_sec", 0.75)
        )
        self.soundcheck_profile_use_loaded_target_corridor = bool(
            autofoh_soundcheck_profile.get("use_loaded_target_corridor", True)
        )
        self.soundcheck_profile_use_phase_target_action_guards = bool(
            autofoh_soundcheck_profile.get("use_phase_target_action_guards", True)
        )
        self.soundcheck_profile_replace_live_target = bool(
            autofoh_soundcheck_profile.get(
                "replace_live_target_with_learned_corridor",
                True,
            )
        )
        self.soundcheck_profile_path = str(
            autofoh_soundcheck_profile.get("path", "") or ""
        )
        lead_masking_config = detector_config.get("lead_masking", {})
        mud_config = detector_config.get("mud_excess", {})
        harshness_config = detector_config.get("harshness_excess", {})
        sibilance_config = detector_config.get("sibilance_excess", {})
        low_end_config = detector_config.get("low_end", {})

        self.mixer_client = None
        self._real_mixer_client = None
        self._observation_mixer_client: Optional[ObservationMixerClient] = None
        self.audio_capture: Optional[AudioCapture] = None
        self.feedback_detector: Optional[FeedbackDetector] = None
        self._last_feedback_event_at: Dict[Tuple[int, str, int], float] = {}
        self._last_feedback_channel_action_at: Dict[int, float] = {}
        self._last_feedback_global_action_at: float = 0.0
        self._feedback_actions_sent_by_channel: Dict[int, int] = {}
        self._feedback_actions_sent_total: int = 0
        self.safety_controller: Optional[AutoFOHSafetyController] = None
        self.decision_engine_v2 = None
        self.decision_engine_v2_safety_gate = None
        self.decision_engine_v2_logger = None
        self.decision_engine_v2_last_result: Optional[Dict[str, Any]] = None
        self.autofoh_logger: Optional[AutoFOHStructuredLogger] = None
        self.autofoh_session_report: Optional[AutoFOHSessionReport] = None
        self.autofoh_session_report_summary: str = ""
        self.perceptual_config = perceptual_config
        self.perceptual_evaluator: Optional[Any] = None
        self._perceptual_pending_audio: Dict[int, Dict[str, Any]] = {}
        self.muq_eval_config = muq_eval_config
        self.muq_eval_service: Optional[Any] = None
        self._muq_pending_audio: Dict[int, Dict[str, Any]] = {}
        self._last_muq_stem_drift: Dict[str, Any] = {}
        self.muq_eval_session_id = f"autofoh_{int(time.time())}"
        if bool(perceptual_config.get("enabled", False)):
            if PerceptualEvaluator is None:
                logger.warning("Perceptual evaluator import failed; shadow evaluation disabled")
            else:
                try:
                    self.perceptual_evaluator = PerceptualEvaluator(perceptual_config)
                except Exception as exc:
                    logger.warning("Perceptual evaluator initialization failed: %s", exc)
                    self.perceptual_evaluator = None
        if bool(muq_eval_config.get("enabled", False)):
            if MuQEvalService is None:
                logger.warning("MuQ-Eval service import failed; quality reward disabled")
            else:
                try:
                    self.muq_eval_service = MuQEvalService(muq_eval_config)
                except Exception as exc:
                    logger.warning("MuQ-Eval service initialization failed: %s", exc)
                    self.muq_eval_service = None
        self.loaded_soundcheck_profile: Optional[AutoFOHSoundcheckProfile] = None
        self.discovered_mixer: Optional[DiscoveredMixer] = None
        self.selected_audio_device: Optional[AudioDevice] = None
        self.audio_devices: List[AudioDevice] = []

        self.state = EngineState.IDLE
        self.runtime_state = RuntimeState.IDLE
        self.channels: Dict[int, ChannelInfo] = {}
        self.bus_snapshots: Dict[int, BusSnapshot] = {}
        self.dca_snapshots: Dict[int, DCASnapshot] = {}
        self.channel_group_routes: Dict[int, Dict[str, List[int]]] = {}
        self._stop_event = threading.Event()
        self._engine_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None

        self.on_state_change = on_state_change
        self.on_channel_update = on_channel_update
        self.on_observation = on_observation

        self._applied_channels: set = set()
        self._analyzers: Dict[int, SignalAnalyzer] = {}
        self._last_autofoh_monitor_analysis_at = 0.0
        self._pending_action_evaluations: List[PendingActionEvaluation] = []
        self._action_evaluation_seq = 0
        self._phase_learning_snapshots: Dict[str, PhaseLearningSnapshot] = {}
        self.current_target_corridor = TargetCorridor.default_intergenre()
        self.lead_masking_analyzer = (
            LeadMaskingAnalyzer(
                masking_threshold_db=float(lead_masking_config.get("masking_threshold_db", 3.0)),
                culprit_share_threshold=float(lead_masking_config.get("min_culprit_contribution", 0.35)),
                persistence_required_cycles=int(lead_masking_config.get("persistence_cycles", 3)),
                lead_boost_db=float(lead_masking_config.get("lead_boost_db", 0.5)),
            )
            if lead_masking_config.get("enabled", True)
            else None
        )
        self.mud_detector = (
            MudExcessDetector(
                threshold_db=float(mud_config.get("threshold_db", 2.5)),
                persistence_required_cycles=int(mud_config.get("persistence_cycles", 3)),
                hysteresis_db=float(mud_config.get("hysteresis_db", 0.75)),
            )
            if mud_config.get("enabled", True)
            else None
        )
        self.harshness_detector = (
            HarshnessExcessDetector(
                threshold_db=float(harshness_config.get("threshold_db", 2.5)),
                persistence_required_cycles=int(harshness_config.get("persistence_cycles", 3)),
                hysteresis_db=float(harshness_config.get("hysteresis_db", 0.75)),
            )
            if harshness_config.get("enabled", True)
            else None
        )
        self.sibilance_detector = (
            SibilanceExcessDetector(
                threshold_db=float(sibilance_config.get("threshold_db", 2.5)),
                persistence_required_cycles=int(sibilance_config.get("persistence_cycles", 3)),
                hysteresis_db=float(sibilance_config.get("hysteresis_db", 0.75)),
            )
            if sibilance_config.get("enabled", True)
            else None
        )
        self.low_end_analyzer = (
            LowEndAnalyzer(
                sub_threshold_db=float(low_end_config.get("sub_threshold_db", 4.0)),
                bass_threshold_db=float(low_end_config.get("bass_threshold_db", 3.0)),
                body_threshold_db=float(low_end_config.get("body_threshold_db", 2.5)),
                culprit_share_threshold=float(low_end_config.get("min_culprit_contribution", 0.35)),
                persistence_required_cycles=int(low_end_config.get("persistence_cycles", 3)),
                hysteresis_db=float(low_end_config.get("hysteresis_db", 0.75)),
            )
            if low_end_config.get("enabled", True)
            else None
        )

        mode = "auto-discover" if auto_discover and not mixer_ip else "manual"
        target = f"{mixer_type or '?'}@{mixer_ip or 'auto'}:{mixer_port or 'auto'}"
        logger.info(
            f"AutoSoundcheckEngine: {mode}, target={target}, "
            f"channels={self._configured_channel_count()}, sr={sample_rate}, "
            f"observe_only={observe_only}, "
            f"min_class_conf={self.minimum_auto_apply_classification_confidence:.2f}, "
            f"preserve_existing_processing={self.preserve_existing_processing}, "
            f"eval_enabled={self.evaluation_policy.enabled}, "
            f"perceptual_enabled={self._perceptual_shadow_enabled()}, "
            f"muq_eval_enabled={self._muq_eval_enabled()}, "
            f"shared_chat_mix={self.shared_chat_mix_config.enabled}, "
            f"log_enabled={self.autofoh_logging_enabled}, "
            f"profile_enabled={self.soundcheck_profile_enabled}"
        )

    def _configured_channel_count(self) -> int:
        if self.selected_channels:
            return len(self.selected_channels)
        return self.num_channels

    def _iter_channels(self) -> List[int]:
        if self.selected_channels:
            return [ch for ch in self.selected_channels if ch <= self.num_channels]
        return list(range(1, self.num_channels + 1))

    def _emit_observation(self, operation: Dict[str, Any] = None, message: str = "", summary: Dict[str, Any] = None):
        if not self.on_observation:
            return
        payload = {}
        if message:
            payload["message"] = message
        if operation is not None:
            payload["operation"] = operation
            payload.setdefault("message", f"[OBSERVE] {operation.get('message', '')}")
        if summary is not None:
            payload["summary"] = summary
        try:
            self.on_observation(payload)
        except Exception:
            pass

    def _default_autofoh_log_path(self) -> Path:
        configured_path = (self.autofoh_log_path or "").strip()
        if configured_path:
            return Path(configured_path)
        session_dir = Path(self.config_manager.get("session.session_dir", "sessions"))
        return session_dir / "autofoh_actions.jsonl"

    def _default_autofoh_report_path(self) -> Path:
        configured_path = (self.autofoh_report_path or "").strip()
        if configured_path:
            return Path(configured_path)
        session_dir = Path(self.config_manager.get("session.session_dir", "sessions"))
        return session_dir / "autofoh_session_report.json"

    def _generate_autofoh_session_report(self) -> Optional[Path]:
        if not self.autofoh_logging_enabled or not self.autofoh_write_session_report_on_stop:
            return None

        log_path = self._default_autofoh_log_path()
        if not log_path.exists():
            return None

        report_path = self._default_autofoh_report_path()
        try:
            report = build_session_report_from_jsonl(log_path)
            write_session_report(report, report_path)
        except Exception as exc:
            logger.warning("Failed to write AutoFOH session report: %s", exc)
            self.autofoh_session_report = None
            self.autofoh_session_report_summary = ""
            return None

        self.autofoh_session_report = report
        self.autofoh_session_report_summary = render_session_report_summary(report)
        if self.autofoh_session_report_summary:
            logger.info(self.autofoh_session_report_summary)
            if self.on_observation:
                self._emit_observation(
                    message=self.autofoh_session_report_summary,
                    summary={
                        "autofoh_report_path": str(report_path),
                        "guard_block_count": report.guard_block_count,
                        "action_blocked_count": report.action_blocked_count,
                    },
                )
        return report_path

    def _default_soundcheck_profile_path(self) -> Path:
        configured_path = (self.soundcheck_profile_path or "").strip()
        if configured_path:
            return Path(configured_path)
        session_dir = Path(self.config_manager.get("session.session_dir", "sessions"))
        return session_dir / "autofoh_soundcheck_profile.json"

    def _soundcheck_profile_store(self) -> AutoFOHSoundcheckProfileStore:
        return AutoFOHSoundcheckProfileStore(self._default_soundcheck_profile_path())

    def _load_soundcheck_profile(self) -> Optional[AutoFOHSoundcheckProfile]:
        if not self.soundcheck_profile_enabled:
            return None
        store = self._soundcheck_profile_store()
        if not store.exists():
            return None
        try:
            profile = store.load()
        except Exception as exc:
            logger.warning("Failed to load AutoFOH soundcheck profile: %s", exc)
            self._log_autofoh_event(
                "soundcheck_profile_load_failed",
                profile_path=str(store.path),
                error=str(exc),
            )
            return None

        self.loaded_soundcheck_profile = profile
        if self.soundcheck_profile_use_loaded_target_corridor:
            self.current_target_corridor = profile.target_corridor
        logger.info(
            "Loaded AutoFOH soundcheck profile '%s' (%s channels)",
            profile.name,
            profile.channel_count,
        )
        self._log_autofoh_event(
            "soundcheck_profile_loaded",
            profile_path=str(store.path),
            profile_name=profile.name,
            channel_count=profile.channel_count,
            target_corridor=profile.target_corridor.to_dict(),
        )
        return profile

    def _save_soundcheck_profile(
        self,
        profile: AutoFOHSoundcheckProfile,
    ) -> Optional[Path]:
        if not self.soundcheck_profile_enabled:
            return None
        store = self._soundcheck_profile_store()
        try:
            store.save(profile)
        except Exception as exc:
            logger.warning("Failed to save AutoFOH soundcheck profile: %s", exc)
            self._log_autofoh_event(
                "soundcheck_profile_save_failed",
                profile_path=str(store.path),
                profile_name=profile.name,
                error=str(exc),
            )
            return None

        self.loaded_soundcheck_profile = profile
        if self.soundcheck_profile_replace_live_target:
            self.current_target_corridor = profile.target_corridor
        logger.info(
            "Saved AutoFOH soundcheck profile '%s' to %s",
            profile.name,
            store.path,
        )
        self._log_autofoh_event(
            "soundcheck_profile_saved",
            profile_path=str(store.path),
            profile_name=profile.name,
            channel_count=profile.channel_count,
            target_corridor=profile.target_corridor.to_dict(),
        )
        return store.path

    def _ensure_autofoh_logger(self) -> Optional[AutoFOHStructuredLogger]:
        if not self.autofoh_logging_enabled:
            return None
        if self.autofoh_logger is None:
            self.autofoh_logger = AutoFOHStructuredLogger(
                path=self._default_autofoh_log_path(),
                queue_maxsize=self.autofoh_log_queue_maxsize,
            )
        if not self.autofoh_logger.is_running:
            self.autofoh_logger.start()
        return self.autofoh_logger

    def _feature_snapshot_payload(self, features) -> Optional[Dict[str, Any]]:
        if features is None:
            return None
        return {
            "rms_db": round(float(features.rms_db), 3),
            "peak_db": round(float(features.peak_db), 3),
            "crest_factor_db": round(float(features.crest_factor_db), 3),
            "mix_indexes": {
                key: round(float(value), 3)
                for key, value in features.mix_indexes.as_dict().items()
            },
            "named_band_levels_db": {
                key: round(float(value), 3)
                for key, value in features.named_band_levels_db.items()
            },
        }

    def _log_autofoh_event(self, event_type: str, **payload):
        logger_instance = self._ensure_autofoh_logger()
        if logger_instance is None:
            return False
        event_payload = {
            "engine_state": self.state.value,
            "runtime_state": self.runtime_state.value,
        }
        event_payload.update(payload)
        return logger_instance.log_event(event_type, **event_payload)

    @staticmethod
    def _safe_call(callback, default=None):
        try:
            value = callback()
            if value is None:
                return default
            return value
        except Exception:
            return default

    def _load_muq_eval_config(self) -> Dict[str, Any]:
        """Load MuQ-Eval config from ConfigManager and optional YAML drift profiles."""

        config = dict(self.config_manager.get_section("muq_eval") or {})
        config_path = Path(__file__).resolve().parents[1] / "config" / "muq_eval.yaml"
        drift_path = Path(__file__).resolve().parents[1] / "config" / "ewma_metrics.yaml"
        try:
            import yaml
        except Exception as exc:
            logger.warning("MuQ-Eval YAML config load skipped: %s", exc)
            return config

        if config_path.exists():
            try:
                with config_path.open(encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or {}
            except Exception as exc:
                logger.warning("MuQ-Eval config load skipped: %s", exc)
            else:
                if isinstance(payload, dict) and isinstance(payload.get("muq_eval"), dict):
                    payload = payload["muq_eval"]
                if isinstance(payload, dict):
                    config.update(payload)

        if drift_path.exists():
            try:
                with drift_path.open(encoding="utf-8") as handle:
                    drift_payload = yaml.safe_load(handle) or {}
            except Exception as exc:
                logger.warning("EWMA drift config load skipped: %s", exc)
            else:
                if isinstance(drift_payload, dict) and isinstance(
                    drift_payload.get("ewma_metrics"),
                    dict,
                ):
                    drift_payload = drift_payload["ewma_metrics"]
                if isinstance(drift_payload, dict):
                    merged_drift = dict(config.get("stem_drift") or {})
                    merged_drift.update(drift_payload)
                    config["stem_drift"] = merged_drift
        return config

    def _perceptual_shadow_enabled(self) -> bool:
        evaluator = getattr(self, "perceptual_evaluator", None)
        return bool(
            evaluator is not None
            and getattr(evaluator, "enabled", False)
            and str(getattr(evaluator, "mode", "shadow")).lower() == "shadow"
        )

    def _muq_eval_enabled(self) -> bool:
        service = getattr(self, "muq_eval_service", None)
        return bool(service is not None and getattr(service, "enabled", False))

    def _muq_shadow_mode(self) -> bool:
        service = getattr(self, "muq_eval_service", None)
        if service is None:
            return True
        return bool(getattr(getattr(service, "config", None), "shadow_mode", True))

    def _capture_muq_master_audio(self) -> Optional[np.ndarray]:
        if not self._muq_eval_enabled():
            return None
        window_sec = float(self.muq_eval_config.get("window_sec", 10.0))
        return self._capture_master_reference_audio(window_sec=window_sec)

    def _muq_action_context(self, action, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"action_type": getattr(action, "action_type", action.__class__.__name__)}
        if hasattr(action, "__dict__"):
            payload.update(action.__dict__)
        if snapshot:
            previous_state = (
                snapshot.get("metadata", {})
                .get("previous_state", {})
            )
            if previous_state:
                payload["previous_state"] = dict(previous_state)
        return payload

    def _is_muq_aggressive_action(self, action) -> bool:
        return isinstance(
            action,
            (
                ChannelGainMove,
                ChannelFaderMove,
                BusFaderMove,
                DCAFaderMove,
                MasterFaderMove,
                ChannelEQMove,
                BusEQMove,
                CompressorAdjust,
                CompressorMakeupAdjust,
                BusCompressorAdjust,
                BusCompressorMakeupAdjust,
            ),
        )

    def _preflight_muq_quality_gate(
        self,
        action,
        snapshot: Dict[str, Any],
        runtime_state: RuntimeState,
    ):
        """Optionally block aggressive writes when MuQ-Eval is in non-shadow mode."""

        if (
            not self._muq_eval_enabled()
            or self._muq_shadow_mode()
            or not self._is_muq_aggressive_action(action)
        ):
            return None
        before_audio = snapshot.get("muq_before_audio")
        try:
            return self.muq_eval_service.validate_correction(
                before_audio=before_audio,
                sample_rate=self.sample_rate,
                proposed_action=self._muq_action_context(action, snapshot),
                after_audio=None,
                session_id=self.muq_eval_session_id,
                current_scene=runtime_state.value,
                osc_commands=[],
                safety_penalty=0.0,
            )
        except Exception as exc:
            logger.warning("MuQ-Eval preflight failed; blocking aggressive action: %s", exc)
            return None

    def _submit_muq_quality_validation(
        self,
        pending: PendingActionEvaluation,
        decision: Optional[SafetyDecision] = None,
        control_state_applied: Optional[bool] = None,
    ) -> Optional[Any]:
        if not self._muq_eval_enabled():
            return None
        pending_audio = self._muq_pending_audio.pop(pending.evaluation_id, None)
        if not pending_audio:
            return None
        after_audio = self._capture_muq_master_audio()
        safety_penalty = 0.0
        if decision is not None:
            if not decision.sent:
                safety_penalty = 0.5
            elif decision.bounded:
                safety_penalty = 0.05
        if control_state_applied is False:
            safety_penalty = max(safety_penalty, 0.25)
        try:
            muq_decision = self.muq_eval_service.validate_correction(
                before_audio=pending_audio.get("before_audio"),
                sample_rate=self.sample_rate,
                proposed_action=pending_audio.get("action") or self._muq_action_context(pending.action),
                after_audio=after_audio,
                session_id=self.muq_eval_session_id,
                current_scene=pending.runtime_state.value,
                osc_commands=pending_audio.get("osc_commands") or [],
                safety_penalty=safety_penalty,
            )
        except Exception as exc:
            logger.warning("MuQ-Eval validation failed: %s", exc)
            return None
        self._log_autofoh_event(
            "muq_quality_decision",
            evaluation_id=pending.evaluation_id,
            channel_id=pending.channel_id,
            action=pending.action,
            quality_decision=muq_decision.to_dict(),
            reward=muq_decision.reward,
        )
        return muq_decision

    def update_muq_stem_score_batch(
        self,
        stem_scores: Dict[str, Any],
        dt: Optional[float] = None,
        params_by_stem: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Update per-stem MuQ EWMA drift metrics for visualization and freeze guards."""

        service = getattr(self, "muq_eval_service", None)
        if service is None or not hasattr(service, "update_stem_score_batch"):
            return {"enabled": False, "stems": {}, "summary": "MuQ stem EWMA drift unavailable"}

        result = service.update_stem_score_batch(
            stem_scores,
            dt,
            params_by_stem=params_by_stem,
        )
        self._last_muq_stem_drift = dict(result or {})
        if not result.get("enabled", False):
            return result

        self._log_autofoh_event("muq_stem_drift", drift=result)
        summary = str(result.get("summary", "MuQ stem EWMA drift updated"))
        attention = [
            f"{stem}:{payload.get('state')}"
            for stem, payload in result.get("stems", {}).items()
            if payload.get("state") in {"WARN", "CRIT"}
        ]
        if attention:
            summary = f"{summary}; attention={', '.join(attention)}"

        self._emit_observation(
            message=summary,
            summary={"muq_stem_drift": result},
            operation={
                "type": "muq_stem_drift",
                "osc_endpoint": self._muq_stem_drift_osc_endpoint(result),
                "stems": result.get("stems", {}),
                "frozen_stems": result.get("frozen_stems", {}),
            },
        )
        return result

    @staticmethod
    def _muq_stem_drift_osc_endpoint(result: Dict[str, Any]) -> str:
        for stem_result in result.get("stems", {}).values():
            endpoint = str(stem_result.get("osc_endpoint", ""))
            if endpoint:
                return endpoint.rsplit("/", 1)[0]
        return "/autofoh/muq_drift"

    def _stem_ml_corrections_frozen(self, info: ChannelInfo) -> bool:
        service = getattr(self, "muq_eval_service", None)
        monitor = getattr(service, "stem_drift_monitor", None)
        if service is None or monitor is None:
            return False

        frozen = getattr(monitor, "frozen_stems", lambda: {})()
        if not frozen:
            return False

        candidates = [
            str(info.source_role or ""),
            str(info.preset or ""),
            *(str(role) for role in (info.stem_roles or [])),
        ]
        candidate_groups = {
            getattr(monitor, "infer_group", lambda value: str(value))(candidate)
            for candidate in candidates
            if candidate
        }
        for frozen_stem in frozen:
            if frozen_stem in candidates:
                return True
            frozen_group = getattr(monitor, "infer_group", lambda value: str(value))(frozen_stem)
            if frozen_group in candidate_groups:
                return True
        return False

    def _capture_perceptual_audio(self, channel_id: Optional[int]) -> Optional[np.ndarray]:
        if (
            not self._perceptual_shadow_enabled()
            or channel_id is None
            or self.audio_capture is None
            or not bool(self.perceptual_config.get("evaluate_channels", True))
        ):
            return None
        window_seconds = float(self.perceptual_config.get("window_seconds", 5.0))
        num_samples = max(self.block_size, int(self.sample_rate * max(0.1, window_seconds)))
        try:
            samples = self.audio_capture.get_buffer(channel_id, num_samples)
        except Exception as exc:
            logger.debug("Perceptual pre/post audio capture failed for ch %s: %s", channel_id, exc)
            return None
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return None
        return samples.copy()

    def _submit_perceptual_shadow_evaluation(
        self,
        pending: PendingActionEvaluation,
        outcome: Optional[ActionEvaluationOutcome] = None,
        control_state_applied: Optional[bool] = None,
    ) -> bool:
        if not self._perceptual_shadow_enabled():
            return False
        pending_audio = self._perceptual_pending_audio.pop(pending.evaluation_id, None)
        if not pending_audio:
            return False
        after_audio = self._capture_perceptual_audio(pending.channel_id)
        if after_audio is None:
            return False

        info = self.channels.get(pending.channel_id) if pending.channel_id is not None else None
        engineering_score = 0.0
        if outcome is not None:
            if outcome.improved:
                engineering_score = 1.0
            elif outcome.worsened:
                engineering_score = -1.0

        action_payload = pending.action
        context = {
            "timestamp": time.time(),
            "channel": info.name if info is not None else str(pending.channel_id),
            "instrument": (
                info.preset
                or info.source_role
                if info is not None
                else None
            ),
            "action": action_payload,
            "engineering_score": engineering_score,
            "safety_score": 1.0 if (control_state_applied is not False) else 0.0,
            "evaluation_id": pending.evaluation_id,
            "runtime_state": pending.runtime_state.value,
            "expected_effect": pending.expected_effect,
            "metadata": pending.metadata,
        }
        try:
            return bool(
                self.perceptual_evaluator.submit_shadow_evaluation(
                    pending_audio["before_audio"],
                    after_audio,
                    self.sample_rate,
                    context=context,
                    osc_sent=bool(pending_audio.get("osc_sent", True)),
                )
            )
        except Exception as exc:
            logger.warning("Perceptual shadow submit failed: %s", exc)
            return False

    def _phase_target_for_runtime_state(self, runtime_state: Optional[RuntimeState] = None):
        profile = self.loaded_soundcheck_profile
        if profile is None or not getattr(profile, "phase_targets", None):
            return None

        effective_state = runtime_state or self.runtime_state
        phase_mapping = {
            RuntimeState.SILENCE_CAPTURE: "SILENCE_CAPTURE",
            RuntimeState.LINE_CHECK: "LINE_CHECK",
            RuntimeState.SOURCE_LEARNING: "SOURCE_LEARNING",
            RuntimeState.STEM_LEARNING: "STEM_LEARNING",
            RuntimeState.FULL_BAND_LEARNING: "FULL_BAND_LEARNING",
            RuntimeState.SNAPSHOT_LOCK: "SNAPSHOT_LOCK",
            RuntimeState.PRE_SHOW_CHECK: "SNAPSHOT_LOCK",
            RuntimeState.LOAD_SONG_SNAPSHOT: "SNAPSHOT_LOCK",
            RuntimeState.SONG_START_STABILIZE: "FULL_BAND_LEARNING",
            RuntimeState.VERSE: "FULL_BAND_LEARNING",
            RuntimeState.CHORUS: "FULL_BAND_LEARNING",
            RuntimeState.SOLO: "FULL_BAND_LEARNING",
            RuntimeState.SPEECH: "FULL_BAND_LEARNING",
            RuntimeState.BETWEEN_SONGS: "SNAPSHOT_LOCK",
            RuntimeState.EMERGENCY_FEEDBACK: "SNAPSHOT_LOCK",
            RuntimeState.EMERGENCY_SPL: "SNAPSHOT_LOCK",
            RuntimeState.EMERGENCY_SIGNAL_LOSS: "SNAPSHOT_LOCK",
            RuntimeState.ROLLBACK: "SNAPSHOT_LOCK",
        }

        candidate_names: List[str] = []
        if effective_state is not None:
            candidate_names.append(effective_state.value)
            mapped_name = phase_mapping.get(effective_state)
            if mapped_name:
                candidate_names.append(mapped_name)
        candidate_names.extend(
            [
                "SNAPSHOT_LOCK",
                "FULL_BAND_LEARNING",
                "STEM_LEARNING",
                "SOURCE_LEARNING",
                "LINE_CHECK",
            ]
        )

        seen = set()
        for phase_name in candidate_names:
            if not phase_name or phase_name in seen:
                continue
            seen.add(phase_name)
            phase_target = profile.phase_targets.get(phase_name)
            if phase_target is not None:
                return phase_target
        return None

    def _target_corridor_for_runtime_state(
        self,
        runtime_state: Optional[RuntimeState] = None,
    ) -> TargetCorridor:
        phase_target = self._phase_target_for_runtime_state(runtime_state)
        if (
            phase_target is not None
            and self.soundcheck_profile_use_loaded_target_corridor
        ):
            return phase_target.target_corridor
        return self.current_target_corridor

    def _build_phase_target_guard_context(
        self,
        *,
        runtime_state: Optional[RuntimeState] = None,
        include_inactive: bool = False,
    ) -> Optional[Dict[str, Any]]:
        channel_features, stem_features, _, channel_stems = self._collect_autofoh_features(
            include_inactive=include_inactive,
            runtime_state=runtime_state or self.runtime_state,
        )
        if not channel_features:
            return None
        return {
            "channel_features": channel_features,
            "stem_features": stem_features,
            "channel_stems": channel_stems,
        }

    @staticmethod
    def _phase_target_guard_enabled_for_action(
        action,
        runtime_state: RuntimeState,
    ) -> bool:
        if runtime_state in {
            RuntimeState.EMERGENCY_FEEDBACK,
            RuntimeState.EMERGENCY_SPL,
            RuntimeState.EMERGENCY_SIGNAL_LOSS,
            RuntimeState.ROLLBACK,
        }:
            return False
        return isinstance(
            action,
            (
                ChannelGainMove,
                ChannelFaderMove,
                ChannelEQMove,
                HighPassAdjust,
                CompressorAdjust,
                SendLevelAdjust,
            ),
        )

    def _capture_live_analysis_features(
        self,
        channel_id: Optional[int],
        runtime_state: Optional[RuntimeState] = None,
    ):
        if channel_id is None or self.audio_capture is None:
            return None
        fft_size = int(self.autofoh_analysis_config.get("fft_size", 4096))
        octave_fraction = int(self.autofoh_analysis_config.get("octave_fraction", 3))
        slope_db = float(
            self.autofoh_analysis_config.get("slope_compensation_db_per_octave", 4.5)
        )
        samples = self.audio_capture.get_buffer(channel_id, fft_size)
        if len(samples) < max(self.block_size, fft_size // 2):
            return None
        return extract_analysis_features(
            samples,
            sample_rate=self.sample_rate,
            fft_size=fft_size,
            octave_fraction=octave_fraction,
            slope_compensation_db_per_octave=slope_db,
            target_corridor=self._target_corridor_for_runtime_state(runtime_state),
        )

    def _expected_effect_for_action(self, action, band_name: Optional[str]) -> str:
        if isinstance(action, ChannelEQMove):
            return f"Move {band_name or 'target'} spectral error closer to corridor"
        if isinstance(action, ChannelFaderMove):
            direction = "raise" if action.delta_db >= 0.0 else "lower"
            return f"{direction} channel level without exceeding safe bounds"
        if isinstance(action, BusEQMove):
            return f"Move bus {action.bus_id} {band_name or 'target'} spectral balance safely"
        if isinstance(action, BusFaderMove):
            direction = "raise" if action.delta_db >= 0.0 else "lower"
            return f"{direction} bus {action.bus_id} level within safe bounds"
        if isinstance(action, DCAFaderMove):
            direction = "raise" if action.delta_db >= 0.0 else "lower"
            return f"{direction} DCA {action.dca_id} level within safe bounds"
        if isinstance(action, DelayAdjust):
            return f"align channel {action.channel_id} timing without resetting existing input delay"
        if isinstance(action, PolarityAdjust):
            return f"set channel {action.channel_id} polarity from measured phase correlation"
        return action.reason

    def _capture_action_snapshot(
        self,
        action,
        evaluation_context: Optional[Dict[str, Any]] = None,
        runtime_state: Optional[RuntimeState] = None,
    ):
        evaluation_context = evaluation_context or {}
        channel_id = getattr(action, "channel_id", None)
        pre_features = evaluation_context.get("pre_features")
        if pre_features is None:
            pre_features = self._capture_live_analysis_features(
                channel_id,
                runtime_state=runtime_state,
            )

        previous_state: Dict[str, Any] = {}
        if self.mixer_client is not None:
            if isinstance(action, ChannelFaderMove):
                previous_state["target_db"] = self._safe_call(
                    lambda: self.mixer_client.get_fader(channel_id),
                    None,
                )
            elif isinstance(action, ChannelEQMove):
                previous_state["gain_db"] = self._safe_call(
                    lambda: self.mixer_client.get_eq_band_gain(channel_id, f"{action.band}g"),
                    None,
                )
                previous_state["freq_hz"] = self._safe_call(
                    lambda: self.mixer_client.get_eq_band_frequency(channel_id, f"{action.band}f"),
                    action.freq_hz,
                )
                previous_state["q"] = action.q
            elif isinstance(action, DelayAdjust):
                previous_state["delay_ms"] = self._safe_call(
                    lambda: self.mixer_client.get_delay(channel_id),
                    None,
                )
            elif isinstance(action, PolarityAdjust):
                previous_state["inverted"] = self._safe_call(
                    lambda: self.mixer_client.get_polarity(channel_id),
                    None,
                )

        evaluation_band = str(
            evaluation_context.get("band_name") or infer_evaluation_band(action) or ""
        ) or None
        rollback_action = build_rollback_action(action, previous_state)
        metadata = {
            key: value
            for key, value in evaluation_context.items()
            if key != "pre_features"
        }
        if previous_state:
            metadata["previous_state"] = dict(previous_state)
        return {
            "channel_id": channel_id,
            "pre_features": pre_features,
            "perceptual_before_audio": self._capture_perceptual_audio(channel_id),
            "muq_before_audio": self._capture_muq_master_audio()
            if self._is_muq_aggressive_action(action)
            else None,
            "rollback_action": rollback_action,
            "evaluation_band_name": evaluation_band,
            "expected_effect": str(
                evaluation_context.get("expected_effect")
                or self._expected_effect_for_action(action, evaluation_band)
            ),
            "metadata": metadata,
        }

    def _register_pending_action_evaluation(
        self,
        action,
        runtime_state: RuntimeState,
        snapshot: Optional[Dict[str, Any]] = None,
    ):
        if (
            not self.evaluation_policy.enabled
            and not self._perceptual_shadow_enabled()
            and not self._muq_eval_enabled()
        ):
            return None
        channel_id = getattr(action, "channel_id", None)
        if channel_id is None and not self._muq_eval_enabled():
            return None
        snapshot = snapshot or {}
        evaluation_delay_sec = (
            float(self.evaluation_policy.evaluation_window_sec)
            if self.evaluation_policy.enabled
            else float(self.perceptual_config.get("hop_seconds", 2.0))
        )
        self._action_evaluation_seq += 1
        pending = PendingActionEvaluation(
            evaluation_id=self._action_evaluation_seq,
            action=action,
            registered_at=time.monotonic(),
            due_at=time.monotonic() + evaluation_delay_sec,
            runtime_state=runtime_state,
            channel_id=channel_id,
            pre_features=snapshot.get("pre_features"),
            rollback_action=snapshot.get("rollback_action"),
            evaluation_band_name=snapshot.get("evaluation_band_name"),
            expected_effect=snapshot.get("expected_effect", ""),
            metadata=dict(snapshot.get("metadata", {})),
        )
        if isinstance(action, ChannelFaderMove):
            previous_target = (
                snapshot.get("metadata", {})
                .get("previous_state", {})
                .get("target_db", action.target_db)
            )
            pending.metadata.setdefault(
                "target_delta_db",
                float(action.target_db) - float(previous_target or 0.0),
            )
        self._pending_action_evaluations.append(pending)
        perceptual_before_audio = snapshot.get("perceptual_before_audio")
        if self._perceptual_shadow_enabled() and perceptual_before_audio is not None:
            self._perceptual_pending_audio[pending.evaluation_id] = {
                "before_audio": perceptual_before_audio,
                "osc_sent": True,
            }
        muq_before_audio = snapshot.get("muq_before_audio")
        if self._muq_eval_enabled() and muq_before_audio is not None:
            self._muq_pending_audio[pending.evaluation_id] = {
                "before_audio": muq_before_audio,
                "action": self._muq_action_context(action, snapshot),
                "osc_commands": [],
            }
        self._log_autofoh_event(
            "action_evaluation_scheduled",
            evaluation_id=pending.evaluation_id,
            channel_id=pending.channel_id,
            action=action,
            runtime_state=runtime_state.value,
            expected_effect=pending.expected_effect,
            evaluation_band_name=pending.evaluation_band_name,
            pre_features=self._feature_snapshot_payload(pending.pre_features),
            metadata=pending.metadata,
            rollback_supported=pending.rollback_action is not None,
        )
        return pending

    def _read_control_state_for_action(self, action) -> Tuple[bool, Dict[str, Any]]:
        if self.mixer_client is None:
            return False, {"note": "no mixer client"}
        if isinstance(action, ChannelFaderMove):
            current_value = self._safe_call(
                lambda: self.mixer_client.get_fader(action.channel_id),
                None,
            )
            if current_value is None:
                return False, {"note": "fader state unavailable"}
            return (
                abs(float(current_value) - float(action.target_db)) <= 0.25,
                {
                    "current_value_db": float(current_value),
                    "target_value_db": float(action.target_db),
                },
            )
        if isinstance(action, ChannelEQMove):
            current_value = self._safe_call(
                lambda: self.mixer_client.get_eq_band_gain(action.channel_id, f"{action.band}g"),
                None,
            )
            if current_value is None:
                return False, {"note": "eq gain state unavailable"}
            return (
                abs(float(current_value) - float(action.gain_db)) <= 0.25,
                {
                    "current_value_db": float(current_value),
                    "target_value_db": float(action.gain_db),
                },
            )
        if isinstance(action, MasterFaderMove):
            current_value = self._safe_call(
                lambda: self.mixer_client.get_main_fader(action.main_id),
                None,
            )
            if current_value is None:
                return False, {"note": "main fader state unavailable"}
            return (
                abs(float(current_value) - float(action.target_db)) <= 0.25,
                {
                    "current_value_db": float(current_value),
                    "target_value_db": float(action.target_db),
                },
            )
        if isinstance(action, DelayAdjust):
            current_value = self._safe_call(
                lambda: self.mixer_client.get_delay(action.channel_id),
                None,
            )
            if current_value is None:
                return False, {"note": "delay state unavailable"}
            return (
                abs(float(current_value) - float(action.delay_ms)) <= 0.05,
                {
                    "current_value_ms": float(current_value),
                    "target_value_ms": float(action.delay_ms),
                },
            )
        if isinstance(action, PolarityAdjust):
            current_value = self._safe_call(
                lambda: self.mixer_client.get_polarity(action.channel_id),
                None,
            )
            if current_value is None:
                return False, {"note": "polarity state unavailable"}
            return (
                bool(current_value) == bool(action.inverted),
                {
                    "current_inverted": bool(current_value),
                    "target_inverted": bool(action.inverted),
                },
            )
        return True, {"note": "control-state verification not implemented for action type"}

    def _evaluate_pending_actions(self, force: bool = False) -> List[ActionEvaluationOutcome]:
        if not self._pending_action_evaluations:
            return []

        now = time.monotonic()
        remaining: List[PendingActionEvaluation] = []
        outcomes: List[ActionEvaluationOutcome] = []

        for pending in self._pending_action_evaluations:
            if not force and pending.due_at > now:
                remaining.append(pending)
                continue

            control_state_applied, control_metrics = self._read_control_state_for_action(
                pending.action
            )

            outcome = None
            if self.evaluation_policy.enabled:
                post_features = None
                if self.evaluation_policy.allow_proxy_audio_evaluation_for_testing:
                    post_features = self._capture_live_analysis_features(
                        pending.channel_id,
                        runtime_state=pending.runtime_state,
                    )

                outcome = evaluate_pending_action(
                    pending,
                    policy=self.evaluation_policy,
                    control_state_applied=control_state_applied,
                    post_features=post_features,
                )
                outcome.metrics.update(control_metrics)
                outcomes.append(outcome)

                self._log_autofoh_event(
                    "action_evaluation",
                    evaluation_id=pending.evaluation_id,
                    channel_id=pending.channel_id,
                    action=pending.action,
                    expected_effect=pending.expected_effect,
                    evaluation_band_name=pending.evaluation_band_name,
                    measured_effect=outcome.measured_effect,
                    evaluated=outcome.evaluated,
                    observable=outcome.observable,
                    improved=outcome.improved,
                    worsened=outcome.worsened,
                    should_rollback=outcome.should_rollback,
                    control_state_applied=outcome.control_state_applied,
                    metrics=outcome.metrics,
                    note=outcome.note,
                    pre_features=self._feature_snapshot_payload(pending.pre_features),
                    post_features=self._feature_snapshot_payload(post_features),
                    metadata=pending.metadata,
                )

            self._submit_perceptual_shadow_evaluation(
                pending,
                outcome=outcome,
                control_state_applied=control_state_applied,
            )
            muq_decision = self._submit_muq_quality_validation(
                pending,
                decision=None,
                control_state_applied=control_state_applied,
            )

            muq_requests_rollback = bool(
                muq_decision is not None
                and getattr(muq_decision, "should_block_osc", False)
                and getattr(muq_decision, "rejection_reason", "") == "quality_drop"
            )
            if (
                (outcome and outcome.should_rollback)
                or muq_requests_rollback
            ) and outcome and outcome.rollback_action is not None:
                rollback_decision = self._execute_action(
                    outcome.rollback_action,
                    runtime_state=RuntimeState.ROLLBACK,
                    register_evaluation=False,
                    evaluation_context={
                        "expected_effect": "Restore previous console state after adverse evaluation",
                        "rollback_of_evaluation_id": pending.evaluation_id,
                    },
                )
                if (
                    rollback_decision is not None
                    and rollback_decision.sent
                    and isinstance(rollback_decision.action, ChannelFaderMove)
                ):
                    self.channels[rollback_decision.action.channel_id].fader_db = (
                        rollback_decision.action.target_db
                    )
                self._log_autofoh_event(
                    "action_rollback",
                    evaluation_id=pending.evaluation_id,
                    channel_id=pending.channel_id,
                    original_action=pending.action,
                    rollback_action=outcome.rollback_action,
                    measured_effect=outcome.measured_effect,
                    rollback_sent=bool(rollback_decision and rollback_decision.sent),
                    rollback_message=rollback_decision.message if rollback_decision else "no decision",
                )

        self._pending_action_evaluations = remaining
        return outcomes

    def _activate_observation_mode(self):
        if not self.observe_only or not self.mixer_client:
            return
        if isinstance(self.mixer_client, ObservationMixerClient):
            return
        self._real_mixer_client = self.mixer_client
        self._observation_mixer_client = ObservationMixerClient(
            self._real_mixer_client,
            on_command=lambda op: self._emit_observation(operation=op),
        )
        self.mixer_client = self._observation_mixer_client
        self._emit_observation(message="[OBSERVE] Engine observation mode active: mixer writes are intercepted.")

    def _set_state(self, new_state: EngineState, message: str = ""):
        old = self.state
        self.state = new_state
        runtime_mapping = {
            EngineState.IDLE: RuntimeState.IDLE,
            EngineState.DISCOVERING: RuntimeState.PREFLIGHT,
            EngineState.CONNECTING: RuntimeState.PREFLIGHT,
            EngineState.SCANNING_AUDIO: RuntimeState.PREFLIGHT,
            EngineState.SCANNING_CHANNELS: RuntimeState.LINE_CHECK,
            EngineState.READING_STATE: RuntimeState.PREFLIGHT,
            EngineState.RESETTING: RuntimeState.SILENCE_CAPTURE,
            EngineState.WAITING_FOR_SIGNAL: RuntimeState.LINE_CHECK,
            EngineState.ANALYZING: RuntimeState.SOURCE_LEARNING,
            EngineState.APPLYING: RuntimeState.SNAPSHOT_LOCK,
            EngineState.RUNNING: RuntimeState.PRE_SHOW_CHECK,
            EngineState.ERROR: RuntimeState.MANUAL_LOCK,
            EngineState.STOPPED: RuntimeState.IDLE,
        }
        self.runtime_state = runtime_mapping.get(new_state, self.runtime_state)
        logger.info(f"Engine state: {old.value} -> {new_state.value} {message}")
        self._log_autofoh_event(
            "state_transition",
            previous_engine_state=old.value,
            new_engine_state=new_state.value,
            runtime_state=self.runtime_state.value,
            message=message,
        )
        if self.on_state_change:
            try:
                self.on_state_change(new_state.value, message)
            except Exception:
                pass

    def _execute_action(
        self,
        action,
        runtime_state: Optional[RuntimeState] = None,
        register_evaluation: bool = True,
        evaluation_context: Optional[Dict[str, Any]] = None,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        if self.safety_controller is None:
            return None
        effective_state = runtime_state or self.runtime_state
        if self._phase_target_guard_enabled_for_action(action, effective_state):
            guard_context = phase_guard_context
            if guard_context is None:
                guard_context = self._build_phase_target_guard_context(
                    runtime_state=effective_state
                )
            if guard_context is not None:
                guard_allowed, guard_message, guard_metadata = self._phase_target_action_guard(
                    action,
                    channel_features=guard_context.get("channel_features", {}),
                    stem_features=guard_context.get("stem_features", {}),
                    channel_stems=guard_context.get("channel_stems", {}),
                    runtime_state=effective_state,
                )
                if not guard_allowed:
                    decision = SafetyDecision(
                        action=action,
                        runtime_state=effective_state,
                        allowed=False,
                        sent=False,
                        message=guard_message,
                        payload={"phase_target_guard": dict(guard_metadata)},
                    )
                    pre_features = guard_context.get("channel_features", {}).get(
                        getattr(action, "channel_id", None)
                    )
                    evaluation_context = evaluation_context or {}
                    expected_effect = str(
                        evaluation_context.get("expected_effect")
                        or self._expected_effect_for_action(
                            action,
                            str(
                                evaluation_context.get("band_name")
                                or infer_evaluation_band(action)
                                or ""
                            )
                            or None,
                        )
                    )
                    evaluation_band_name = str(
                        evaluation_context.get("band_name")
                        or infer_evaluation_band(action)
                        or ""
                    ) or None
                    self._log_autofoh_event(
                        "phase_target_guard_blocked",
                        channel_id=getattr(action, "channel_id", None),
                        action=action,
                        runtime_state=effective_state.value,
                        message=guard_message,
                        metadata=guard_metadata,
                    )
                    self._log_autofoh_event(
                        "action_decision",
                        channel_id=getattr(action, "channel_id", None),
                        requested_action=action,
                        applied_action=action,
                        requested_runtime_state=effective_state.value,
                        sent=False,
                        allowed=False,
                        supported=True,
                        bounded=False,
                        rate_limited=False,
                        message=guard_message,
                        expected_effect=expected_effect,
                        evaluation_band_name=evaluation_band_name,
                        pre_features=self._feature_snapshot_payload(pre_features),
                        metadata=guard_metadata,
                    )
                    return decision
        snapshot = self._capture_action_snapshot(
            action,
            evaluation_context,
            effective_state,
        )
        muq_gate = self._preflight_muq_quality_gate(action, snapshot, effective_state)
        if muq_gate is not None and getattr(muq_gate, "should_block_osc", False):
            decision = SafetyDecision(
                action=action,
                runtime_state=effective_state,
                allowed=False,
                sent=False,
                message=f"MuQ-Eval quality gate rejected: {muq_gate.rejection_reason}",
                payload={"muq_eval": muq_gate.to_dict()},
            )
            self._log_autofoh_event(
                "action_decision",
                channel_id=getattr(action, "channel_id", None),
                requested_action=action,
                applied_action=action,
                requested_runtime_state=effective_state.value,
                sent=False,
                allowed=False,
                supported=True,
                bounded=False,
                rate_limited=False,
                message=decision.message,
                expected_effect=snapshot.get("expected_effect"),
                evaluation_band_name=snapshot.get("evaluation_band_name"),
                pre_features=self._feature_snapshot_payload(snapshot.get("pre_features")),
                metadata={
                    **dict(snapshot.get("metadata", {})),
                    "muq_quality_decision": muq_gate.to_dict(),
                },
            )
            return decision
        decision = self.safety_controller.execute(action, effective_state)
        self._log_autofoh_event(
            "action_decision",
            channel_id=getattr(decision.action, "channel_id", None),
            requested_action=action,
            applied_action=decision.action,
            requested_runtime_state=effective_state.value,
            sent=decision.sent,
            allowed=decision.allowed,
            supported=decision.supported,
            bounded=decision.bounded,
            rate_limited=decision.rate_limited,
            message=decision.message,
            expected_effect=snapshot.get("expected_effect"),
            evaluation_band_name=snapshot.get("evaluation_band_name"),
            pre_features=self._feature_snapshot_payload(snapshot.get("pre_features")),
            metadata=snapshot.get("metadata", {}),
        )
        if decision.sent and register_evaluation:
            pending = self._register_pending_action_evaluation(
                decision.action,
                effective_state,
                snapshot,
            )
            if pending is not None and pending.evaluation_id in self._muq_pending_audio:
                self._muq_pending_audio[pending.evaluation_id]["osc_commands"] = [
                    dict(decision.payload)
                ] if decision.payload else []
        return decision

    def _ensure_decision_engine_v2(self) -> bool:
        """Initialize the opt-in v2 decision stack."""
        if not self.use_decision_engine_v2:
            return False
        if DecisionEngine is None or SafetyGate is None or MixingKnowledgeBase is None:
            logger.error("Decision Engine v2 imports failed; falling back to legacy path")
            self.use_decision_engine_v2 = False
            return False
        if self.decision_engine_v2 is not None:
            return True
        config = self.decision_engine_v2_config
        try:
            knowledge_path = getattr(config, "knowledge_path", "") if config is not None else ""
            if knowledge_path:
                path = Path(knowledge_path).expanduser()
                if not path.is_absolute():
                    path = Path(__file__).resolve().parents[1] / path
                knowledge = MixingKnowledgeBase.load(path)
            else:
                knowledge = MixingKnowledgeBase.load()
            self.decision_engine_v2 = DecisionEngine(
                knowledge,
                getattr(config, "decision", None),
            )
            self.decision_engine_v2_safety_gate = SafetyGate(
                getattr(config, "safety", None)
            )
            if HumanDecisionLogger is not None and config is not None:
                self.decision_engine_v2_logger = HumanDecisionLogger(config.log_path)
            logger.info(
                "Decision Engine v2 initialized (dry_run=%s)",
                self.decision_engine_v2_dry_run,
            )
            return True
        except Exception as exc:
            logger.error("Decision Engine v2 initialization failed: %s", exc)
            self.use_decision_engine_v2 = False
            return False

    def _build_decision_engine_v2_analyzer_output(self) -> Dict[str, Any]:
        channels = []
        for ch, info in sorted(self.channels.items()):
            metrics = {
                "channel_id": ch,
                "lufs": float(info.lufs),
                "target_lufs": float(INSTRUMENT_TARGET_LUFS.get(info.preset or "custom", -23.0)),
                "peak_db": float(info.peak_db),
                "rms_db": float(info.rms_db),
                "confidence": float(info.classification_confidence),
            }
            if info.metrics is not None:
                metrics.update(info.metrics.to_dict())
                metrics.update({
                    "true_peak_dbtp": float(info.metrics.level.true_peak_dbtp),
                    "crest_factor_db": float(info.metrics.level.crest_factor_db),
                    "dynamic_range_db": float(info.metrics.dynamics.dynamic_range_db),
                    "mud_db": float(info.metrics.spectral.mud_ratio) * 10.0,
                    "harshness_db": float(info.metrics.spectral.presence_ratio) * 10.0,
                })
            channels.append({
                "channel_id": ch,
                "name": info.name,
                "role": info.source_role or info.preset or "unknown",
                "preset": info.preset or "custom",
                "metrics": metrics,
                "confidence": float(info.classification_confidence),
                "source_module": "auto_soundcheck_engine",
            })
        return {
            "source_module": "auto_soundcheck_engine",
            "channels": channels,
        }

    def _build_decision_engine_v2_current_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for ch, info in self.channels.items():
            snapshot = info.original_snapshot
            raw = dict(snapshot.raw_settings or {}) if snapshot is not None else {}
            state[f"channel:{ch}"] = {
                "fader_db": float(snapshot.fader_db) if snapshot is not None else -144.0,
                "pan": self._as_float(raw.get("pan"), 0.0),
                "true_peak_dbtp": (
                    float(info.metrics.level.true_peak_dbtp)
                    if info.metrics is not None
                    else float(info.peak_db)
                ),
                "compression_threshold_db": self._as_float(
                    raw.get("compressor_threshold_db"),
                    -18.0,
                ),
            }
        return state

    def _run_decision_engine_v2(self):
        """Run the opt-in v2 plan/gate/executor layer."""
        if not self._ensure_decision_engine_v2():
            if self.auto_apply:
                logger.warning("Decision Engine v2 unavailable; using legacy correction path")
                return self._apply_corrections()
            return None

        analyzer_output = self._build_decision_engine_v2_analyzer_output()
        current_state = self._build_decision_engine_v2_current_state()
        plan = self.decision_engine_v2.create_action_plan(
            analyzer_output,
            critic_evaluations={},
            mode="live",
        )
        for decision in plan.decisions:
            logger.info("Decision Engine v2 proposed: %s", decision.reason)
            if self.decision_engine_v2_logger is not None:
                self.decision_engine_v2_logger.log_decision(decision)

        executor = ActionPlanExecutor(
            self.mixer_client,
            self.decision_engine_v2_safety_gate,
            logger=self.decision_engine_v2_logger,
            dry_run=self.decision_engine_v2_dry_run or not self.auto_apply,
        )
        result = executor.execute(
            plan,
            current_state=current_state,
            live_mode=True,
        )
        self.decision_engine_v2_last_result = result.to_dict()
        logger.info(
            "Decision Engine v2 complete: sent=%s recommended=%s blocked=%s dry_run=%s",
            len(result.sent),
            len(result.recommended_only),
            len(result.blocked),
            result.safety.dry_run,
        )
        self._log_autofoh_event(
            "decision_engine_v2_result",
            sent_count=len(result.sent),
            recommended_count=len(result.recommended_only),
            blocked_count=len(result.blocked),
            dry_run=result.safety.dry_run,
            result=result.to_dict(),
        )
        if self.soundcheck_profile_capture_multiphase_learning and not result.sent:
            self._capture_learning_phase(
                "SNAPSHOT_LOCK",
                RuntimeState.SNAPSHOT_LOCK,
                metadata={
                    "applied_channel_ids": [],
                    "decision_engine_v2": True,
                    "dry_run": result.safety.dry_run,
                },
                notes="Snapshot lock captured after Decision Engine v2 dry-run/no-send pass",
            )
        return result

    def apply_bus_fader_target(self, bus_id: int, target_db: float, reason: str = "bus level correction"):
        """Apply a bounded bus fader correction through the safety layer."""
        return self._execute_action(
            BusFaderMove(bus_id=bus_id, target_db=target_db, reason=reason),
            register_evaluation=False,
        )

    def apply_dca_fader_target(self, dca_id: int, target_db: float, reason: str = "DCA level correction"):
        """Apply a bounded DCA fader correction through the safety layer."""
        return self._execute_action(
            DCAFaderMove(dca_id=dca_id, target_db=target_db, reason=reason),
            register_evaluation=False,
        )

    def apply_bus_eq(
        self,
        bus_id: int,
        band: int,
        freq_hz: float,
        gain_db: float,
        q: float,
        reason: str = "bus EQ correction",
    ):
        """Apply a bounded bus EQ correction through the safety layer."""
        return self._execute_action(
            BusEQMove(
                bus_id=bus_id,
                band=band,
                freq_hz=freq_hz,
                gain_db=gain_db,
                q=q,
                reason=reason,
            ),
            register_evaluation=False,
        )

    def apply_bus_compressor(
        self,
        bus_id: int,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_db: float = 0.0,
        enabled: bool = True,
        reason: str = "bus compressor correction",
    ):
        """Apply a bounded bus compressor correction through the safety layer."""
        return self._execute_action(
            BusCompressorAdjust(
                bus_id=bus_id,
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
                makeup_db=makeup_db,
                enabled=enabled,
                reason=reason,
            ),
            register_evaluation=False,
        )

    def apply_bus_compressor_makeup(
        self,
        bus_id: int,
        makeup_db: float,
        reason: str = "bus compressor makeup correction",
    ):
        """Apply bounded bus compressor make-up through the safety layer."""
        return self._execute_action(
            BusCompressorMakeupAdjust(
                bus_id=bus_id,
                makeup_db=makeup_db,
                reason=reason,
            ),
            register_evaluation=False,
        )

    def _determine_auto_corrections_enabled(self, info: ChannelInfo) -> bool:
        if not info.allowed_controls:
            return False
        if self.new_or_unknown_channel_auto_corrections_enabled:
            return True
        return (
            info.recognized
            and info.source_role != "unknown"
            and info.classification_confidence >= self.minimum_auto_apply_classification_confidence
        )

    def _apply_classification_to_info(self, info: ChannelInfo, result: Dict[str, Any]):
        info.preset = result.get("preset")
        info.source_role = str(result.get("source_role", "unknown"))
        info.stem_roles = list(result.get("stem_roles", []))
        info.allowed_controls = list(result.get("allowed_controls", []))
        info.priority = float(result.get("priority", 0.0))
        info.classification_confidence = float(result.get("confidence", 0.0))
        info.classification_match_type = str(result.get("match_type", "unknown"))
        info.recognized = bool(result.get("recognized", False))
        info.auto_corrections_enabled = self._determine_auto_corrections_enabled(info)
        self._apply_channel_role_overrides(info)

    def _apply_channel_role_overrides(self, info: ChannelInfo):
        """Apply operator-declared roles that should not be treated as sources."""
        if info.channel in self.master_reference_channels:
            info.preset = "playback"
            info.source_role = "mix_bus_reference"
            info.stem_roles = ["MASTER"]
            info.allowed_controls = []
            info.priority = 0.0
            info.recognized = True
            info.auto_corrections_enabled = False

    def _control_allowed(self, info: ChannelInfo, control_name: str) -> bool:
        if control_name in {"feedback_notch", "emergency_fader"}:
            return control_name in info.allowed_controls
        if self._stem_ml_corrections_frozen(info):
            return False
        return info.auto_corrections_enabled and control_name in info.allowed_controls

    def _log_processing_skip(self, ch: int, info: ChannelInfo, stage: str):
        logger.info(
            "Ch %s '%s': %s skipped (role=%s, conf=%.2f, controls=%s)",
            ch,
            info.name,
            stage,
            info.source_role,
            info.classification_confidence,
            ",".join(info.allowed_controls) if info.allowed_controls else "none",
        )

    # ── 0. Auto-discover mixer ───────────────────────────────────

    def _discover_mixer(self) -> bool:
        """Scan the network for available mixers and select the best one."""
        self._set_state(EngineState.DISCOVERING, "Scanning network for mixers...")

        discovered = discover_mixer_auto(
            preferred_type=self.mixer_type,
            preferred_ip=self.mixer_ip,
            scan_subnet=self.scan_subnet,
            timeout=2.0,
        )

        if discovered is None:
            logger.warning("No mixer found on the network")
            return False

        self.discovered_mixer = discovered
        self.mixer_type = discovered.mixer_type
        self.mixer_ip = discovered.ip
        self.mixer_port = discovered.port
        self.mixer_tls = discovered.tls

        logger.info(
            f"Auto-discovered: {discovered.mixer_type.upper()} "
            f"@ {discovered.ip}:{discovered.port} "
            f"('{discovered.name}', {discovered.discovery_method}, "
            f"{discovered.response_time_ms:.0f}ms)"
        )
        return True

    # ── 1. Connect to mixer ──────────────────────────────────────

    def _connect_mixer(self) -> bool:
        """Connect to the mixer console.

        If auto_discover is True and no IP is specified, runs network
        discovery first to find the mixer automatically.
        """
        need_discovery = self.auto_discover and (
            self.mixer_ip is None or self.mixer_type is None
        )

        if need_discovery:
            if not self._discover_mixer():
                # Discovery failed — try known defaults as last resort
                if self.mixer_ip is None:
                    self._set_state(
                        EngineState.ERROR,
                        "No mixer found on the network and no IP specified",
                    )
                    return False
        elif self.auto_discover and self.mixer_ip:
            # Have a preferred IP but still try discovery to confirm type
            self._set_state(EngineState.DISCOVERING, f"Probing {self.mixer_ip}...")
            discovered = discover_mixer_auto(
                preferred_type=self.mixer_type,
                preferred_ip=self.mixer_ip,
                scan_subnet=False,
                timeout=2.0,
            )
            if discovered:
                self.discovered_mixer = discovered
                self.mixer_type = discovered.mixer_type
                self.mixer_ip = discovered.ip
                self.mixer_port = discovered.port
                self.mixer_tls = discovered.tls
                logger.info(f"Confirmed mixer at {discovered.ip}: {discovered.mixer_type.upper()}")
            else:
                logger.info(
                    f"Preferred IP {self.mixer_ip} not responding to probes, "
                    f"trying direct connect..."
                )

        # Ensure we have a type and port
        if self.mixer_type is None:
            self.mixer_type = "dlive"
        if self.mixer_port is None:
            if self.mixer_type == "dlive":
                self.mixer_port = 51328
            else:
                self.mixer_port = 2223

        self._set_state(
            EngineState.CONNECTING,
            f"Connecting to {self.mixer_type.upper()} @ {self.mixer_ip}:{self.mixer_port}...",
        )

        try:
            if self.mixer_type == "dlive":
                from dlive_client import DLiveClient
                self.mixer_client = DLiveClient(
                    ip=self.mixer_ip,
                    port=self.mixer_port,
                    tls=self.mixer_tls,
                    midi_base_channel=self.midi_base_channel,
                )
            elif self.mixer_type == "wing":
                from wing_client import WingClient
                self.mixer_client = WingClient(
                    ip=self.mixer_ip,
                    port=self.mixer_port,
                )
            else:
                logger.error(f"Unsupported mixer type: {self.mixer_type}")
                return False

            success = self.mixer_client.connect(timeout=10.0)
            if success:
                logger.info(f"Connected to {self.mixer_type.upper()} at {self.mixer_ip}:{self.mixer_port}")
                self._activate_observation_mode()
                self.safety_controller = AutoFOHSafetyController(
                    mixer_client=self.mixer_client,
                    config=self.action_safety_config,
                )
                return True
            else:
                logger.error(f"Failed to connect to {self.mixer_type.upper()}")
                return False

        except Exception as e:
            logger.error(f"Mixer connection error: {e}")
            return False

    # ── 2. Detect & start audio ──────────────────────────────────

    def _start_audio(self) -> bool:
        """Scan for audio devices, select the best one, and start capture."""
        self._set_state(EngineState.SCANNING_AUDIO, "Scanning audio devices...")

        # Step 1: Scan all available audio input devices
        self.audio_devices = scan_audio_devices()

        if not self.audio_devices:
            logger.warning("No audio input devices found via scan")

        # Step 2: Select the best device
        preferred_proto = None
        if self.audio_device_name:
            name_lower = self.audio_device_name.lower()
            if "soundgrid" in name_lower or "waves" in name_lower:
                preferred_proto = AudioProtocol.SOUNDGRID
            elif "dante" in name_lower:
                preferred_proto = AudioProtocol.DANTE

        best = select_best_device(
            self.audio_devices,
            preferred_protocol=preferred_proto,
            preferred_name=self.audio_device_name,
            min_channels=2,
        )

        device_index = None
        source_type = AudioSourceType.SOUNDDEVICE

        if best:
            self.selected_audio_device = best
            device_index = best.index
            num_hw_channels = best.max_input_channels

            # Map protocol to source type
            proto_source_map = {
                AudioProtocol.SOUNDGRID: AudioSourceType.SOUNDGRID,
                AudioProtocol.DANTE: AudioSourceType.DANTE,
            }
            source_type = proto_source_map.get(best.protocol, AudioSourceType.SOUNDDEVICE)

            # Adjust channel count to what the device actually supports
            if num_hw_channels < self.num_channels:
                logger.info(
                    f"Device '{best.name}' has {num_hw_channels}ch, "
                    f"reducing capture from {self.num_channels} to {num_hw_channels}"
                )
                self.num_channels = num_hw_channels
                if self.selected_channels:
                    before = list(self.selected_channels)
                    self.selected_channels = [ch for ch in self.selected_channels if ch <= self.num_channels]
                    removed = sorted(set(before) - set(self.selected_channels))
                    if removed:
                        logger.warning(
                            "Selected channels exceed device channel count and will be skipped: %s",
                            removed,
                        )

            logger.info(
                f"Audio device selected: [{best.index}] '{best.name}' "
                f"({best.max_input_channels}ch, {best.protocol.value}, "
                f"score={best.score})"
            )

            device_sample_rate = getattr(best, "default_samplerate", None)
            if device_sample_rate:
                selected_sample_rate = int(device_sample_rate)
                if selected_sample_rate != self.sample_rate:
                    logger.info(
                        "Using selected audio device sample rate: %sHz "
                        "(configured %sHz)",
                        selected_sample_rate,
                        self.sample_rate,
                    )
                    self.sample_rate = selected_sample_rate
        else:
            logger.warning("No suitable audio device found, using system default")

        # Log all discovered devices for diagnostics
        if self.audio_devices:
            logger.info(f"All audio devices ({len(self.audio_devices)}):")
            for dev in self.audio_devices:
                marker = " <<<" if dev == best else ""
                logger.info(
                    f"  [{dev.index}] {dev.name} | {dev.max_input_channels}ch | "
                    f"{dev.protocol.value} | score={dev.score}{marker}"
                )

        # Step 3: Create AudioCapture and start
        self.audio_capture = AudioCapture(
            num_channels=self.num_channels,
            sample_rate=self.sample_rate,
            buffer_seconds=self.buffer_seconds,
            block_size=self.block_size,
            source_type=source_type,
            device_name=device_index,
        )

        try:
            self.audio_capture.start()
            dev_name = best.name if best else "system default"
            logger.info(
                f"Audio capture started: '{dev_name}', "
                f"{source_type.value}, {self.num_channels}ch"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False

    # ── 3. Scan channel names ────────────────────────────────────

    def _scan_channels(self):
        """Read channel names from mixer and recognize instruments."""
        self._set_state(EngineState.SCANNING_CHANNELS, "Reading channel names...")

        channel_names: Dict[int, str] = {}

        for ch in self._iter_channels():
            try:
                name = self.mixer_client.get_channel_name(ch)
                channel_names[ch] = name
            except Exception:
                channel_names[ch] = f"Ch {ch}"

        recognition = scan_and_recognize(
            channel_names,
            classifier_config=self.classifier_config,
        )

        for ch in self._iter_channels():
            info = ChannelInfo(channel=ch)
            info.name = channel_names.get(ch, f"Ch {ch}")

            rec = recognition.get(ch, {})
            self._apply_classification_to_info(info, rec)
            self.channels[ch] = info

        recognized = sum(1 for c in self.channels.values() if c.recognized)
        auto_apply_ready = sum(1 for c in self.channels.values() if c.auto_corrections_enabled)
        logger.info(
            "Channel scan complete: %s/%s recognized, %s eligible for auto-correction",
            recognized,
            len(self.channels),
            auto_apply_ready,
        )

    # ── 3b. Read existing channel settings from mixer ────────────

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "on", "yes"}
        return bool(value)

    @staticmethod
    def _bounded_toward(current: float, target: float, max_delta: float) -> float:
        delta = target - current
        delta = max(-max_delta, min(max_delta, delta))
        return current + delta

    def _snapshot_raw(self, info: ChannelInfo) -> Dict[str, Any]:
        if info.original_snapshot is None:
            return {}
        return dict(info.original_snapshot.raw_settings or {})

    def _snapshot_value(self, info: ChannelInfo, key: str, default: Any = None) -> Any:
        return self._snapshot_raw(info).get(key, default)

    def _channel_has_existing_processing(self, raw_settings: Dict[str, Any]) -> bool:
        fader_db = self._as_float(raw_settings.get("fader_db"), -144.0)
        muted = self._as_bool(raw_settings.get("muted"), False)
        gain_db = self._as_float(raw_settings.get("gain_db"), 0.0)
        pan = self._as_float(raw_settings.get("pan"), 0.0)
        hpf_enabled = self._as_bool(raw_settings.get("hpf_enabled"), False)
        hpf_freq = self._as_float(raw_settings.get("hpf_freq"), 20.0)
        compressor_enabled = self._as_bool(raw_settings.get("compressor_enabled"), False)
        gate_enabled = self._as_bool(raw_settings.get("gate_enabled"), False)
        polarity_inverted = self._as_bool(raw_settings.get("polarity_inverted"), False)
        delay_enabled = self._as_bool(raw_settings.get("delay_enabled"), False)
        delay_ms = self._as_float(raw_settings.get("delay_ms"), 0.0)

        if fader_db > -90.0 or (not muted and fader_db > -50.0):
            return True
        if abs(gain_db) > 0.1 or abs(pan) > 0.5:
            return True
        if hpf_enabled and hpf_freq > 25.0:
            return True
        if gate_enabled or compressor_enabled or polarity_inverted or (delay_enabled and abs(delay_ms) > 0.01):
            return True

        eq_bands = raw_settings.get("eq_bands") or []
        for band in eq_bands:
            if len(band) >= 2 and abs(self._as_float(band[1], 0.0)) > 0.1:
                return True
        return False

    def _extract_bus_ids_from_channel_settings(self, raw_settings: Dict[str, Any]) -> List[int]:
        bus_ids = []
        sends = raw_settings.get("active_bus_sends")
        if not sends:
            sends = [
                send for send in raw_settings.get("sends", [])
                if self._as_bool(send.get("active"), False)
            ]
        for send in sends or []:
            try:
                bus_id = int(send.get("bus"))
            except (TypeError, ValueError):
                continue
            if 1 <= bus_id <= 16 and bus_id not in bus_ids:
                bus_ids.append(bus_id)
        return bus_ids

    def _parse_dca_tags(self, tags: Any) -> List[int]:
        dca_ids = set()

        def visit(item: Any):
            if item is None:
                return
            if isinstance(item, (list, tuple, set)):
                for nested in item:
                    visit(nested)
                return
            for token in str(item).replace(",", " ").replace(";", " ").split():
                token = token.strip().upper()
                if not token.startswith("#D"):
                    continue
                suffix = token[2:]
                if suffix.isdigit():
                    dca = int(suffix)
                    if 1 <= dca <= 16:
                        dca_ids.add(dca)

        visit(tags)
        return sorted(dca_ids)

    def _extract_dca_ids_from_settings(self, raw_settings: Dict[str, Any]) -> List[int]:
        dca_ids = []
        for item in raw_settings.get("dca_assignments") or []:
            try:
                dca = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= dca <= 16 and dca not in dca_ids:
                dca_ids.append(dca)
        for dca in self._parse_dca_tags(raw_settings.get("tags")):
            if dca not in dca_ids:
                dca_ids.append(dca)
        return sorted(dca_ids)

    def _is_monitor_bus(self, bus_id: int) -> bool:
        return int(bus_id) in self.monitor_bus_ids

    def _is_effect_bus(self, bus_id: int, bus_name: str = "") -> bool:
        name = str(bus_name or "").lower()
        return int(bus_id) in self.effect_bus_ids or "delay" in name or "reverb" in name

    def _read_group_state_for_channels(self, channels: Optional[List[int]] = None):
        """Read BUS/DCA state connected to the channels under correction."""
        if self.mixer_client is None:
            return

        selected = channels if channels is not None else self._iter_channels()
        bus_sources: Dict[int, List[int]] = {}
        dca_source_channels: Dict[int, List[int]] = {}

        for ch in selected:
            info = self.channels.get(ch)
            if info is None or info.original_snapshot is None:
                continue
            raw_settings = dict(info.original_snapshot.raw_settings or {})
            bus_ids = self._extract_bus_ids_from_channel_settings(raw_settings)
            dca_ids = self._extract_dca_ids_from_settings(raw_settings)
            self.channel_group_routes[ch] = {
                "bus_ids": bus_ids,
                "dca_ids": dca_ids,
            }
            for bus_id in bus_ids:
                bus_sources.setdefault(bus_id, [])
                if ch not in bus_sources[bus_id]:
                    bus_sources[bus_id].append(ch)
            for dca_id in dca_ids:
                dca_source_channels.setdefault(dca_id, [])
                if ch not in dca_source_channels[dca_id]:
                    dca_source_channels[dca_id].append(ch)

        if channels is None:
            self.bus_snapshots = {}
            self.dca_snapshots = {}

        dca_source_buses: Dict[int, List[int]] = {}
        bus_reader = getattr(self.mixer_client, "get_bus_settings", None)
        for bus_id, source_channels in sorted(bus_sources.items()):
            try:
                raw_bus = bus_reader(bus_id) if callable(bus_reader) else {}
            except Exception as exc:
                logger.debug(f"Bus {bus_id}: could not read settings: {exc}")
                raw_bus = {}
            bus_dcas = self._extract_dca_ids_from_settings(raw_bus)
            for dca_id in bus_dcas:
                dca_source_buses.setdefault(dca_id, [])
                if bus_id not in dca_source_buses[dca_id]:
                    dca_source_buses[dca_id].append(bus_id)
            existing_bus = self.bus_snapshots.get(bus_id)
            merged_source_channels = sorted(
                set(source_channels)
                | set(existing_bus.source_channels if existing_bus is not None else [])
            )
            self.bus_snapshots[bus_id] = BusSnapshot(
                bus=bus_id,
                name=str(raw_bus.get("name") or ""),
                fader_db=self._as_float(raw_bus.get("fader_db"), -100.0),
                muted=self._as_bool(raw_bus.get("muted"), False),
                eq_bands=raw_bus.get("eq_bands"),
                compressor_enabled=self._as_bool(raw_bus.get("compressor_enabled"), False),
                dca_assignments=bus_dcas,
                source_channels=merged_source_channels,
                raw_settings=raw_bus,
            )

        dca_ids = set(dca_source_channels.keys()) | set(dca_source_buses.keys())
        dca_reader = getattr(self.mixer_client, "get_dca_settings", None)
        for dca_id in sorted(dca_ids):
            try:
                raw_dca = dca_reader(dca_id) if callable(dca_reader) else {}
            except Exception as exc:
                logger.debug(f"DCA {dca_id}: could not read settings: {exc}")
                raw_dca = {}
            existing_dca = self.dca_snapshots.get(dca_id)
            merged_source_channels = sorted(
                set(dca_source_channels.get(dca_id, []))
                | set(existing_dca.source_channels if existing_dca is not None else [])
            )
            merged_source_buses = sorted(
                set(dca_source_buses.get(dca_id, []))
                | set(existing_dca.source_buses if existing_dca is not None else [])
            )
            self.dca_snapshots[dca_id] = DCASnapshot(
                dca=dca_id,
                name=str(raw_dca.get("name") or ""),
                fader_db=self._as_float(raw_dca.get("fader_db"), -100.0),
                muted=self._as_bool(raw_dca.get("muted"), False),
                source_channels=merged_source_channels,
                source_buses=merged_source_buses,
                raw_settings=raw_dca,
            )

        if bus_sources or dca_ids:
            logger.info(
                "Group state read: %s bus(es), %s DCA group(s) connected to corrected channels",
                len(bus_sources),
                len(dca_ids),
            )

    def _read_channel_state(self):
        """Read current processing settings for all channels.

        Determines which channels already have non-default settings so later
        actions can be relative corrections instead of destructive preset writes.
        """
        self._set_state(
            EngineState.READING_STATE,
            "Reading existing channel settings from mixer..."
        )

        channels_with_processing = 0

        for ch in self._iter_channels():
            info = self.channels.get(ch)
            if info is None:
                continue

            try:
                raw_settings = self.mixer_client.get_channel_settings(ch)
            except Exception as e:
                logger.debug(f"Ch {ch}: could not read settings: {e}")
                raw_settings = {}

            fader_db = self._as_float(raw_settings.get("fader_db"), -144.0)
            muted = self._as_bool(raw_settings.get("muted"), False)
            gain_db = self._as_float(raw_settings.get("gain_db"), 0.0)
            hpf_freq = self._as_float(raw_settings.get("hpf_freq"), 20.0)
            hpf_enabled = self._as_bool(raw_settings.get("hpf_enabled"), False)
            eq_bands = raw_settings.get("eq_bands")
            has_processing = self._channel_has_existing_processing(raw_settings)

            snapshot = ChannelSnapshot(
                channel=ch,
                fader_db=fader_db,
                muted=muted,
                eq_bands=eq_bands,
                hpf_freq=hpf_freq,
                hpf_enabled=hpf_enabled,
                gain_db=gain_db,
                had_processing=has_processing,
                raw_settings=raw_settings,
            )
            info.original_snapshot = snapshot
            info.fader_db = fader_db

            if has_processing:
                channels_with_processing += 1
                logger.debug(
                    f"Ch {ch} '{info.name}': existing settings detected "
                    f"(fader={fader_db:.1f}dB, gain={gain_db:.1f}dB, "
                    f"hpf={hpf_freq:.0f}Hz/{hpf_enabled}, muted={muted})"
                )

        logger.info(
            f"Channel state read: {channels_with_processing}/{len(self.channels)} "
            f"channels have existing settings"
        )
        self._read_group_state_for_channels()

    # ── 3c. Reset channels to neutral before analysis ────────────

    def _reset_channels(self):
        """Prepare channels for analysis without erasing operator processing.

        Destructive reset is opt-in only. The live default preserves current
        EQ, filters, dynamics, delay, polarity, faders, and sends; later stages
        apply bounded corrections relative to the snapshot read in
        _read_channel_state().
        """
        if self.preserve_existing_processing or not self.allow_destructive_reset:
            self._set_state(
                EngineState.RESETTING,
                "Preserving current channel processing before analysis..."
            )

            preserved_count = 0
            skipped_count = 0
            for ch in self._iter_channels():
                info = self.channels.get(ch)
                if info is None:
                    continue
                if not info.auto_corrections_enabled:
                    skipped_count += 1
                    self._log_processing_skip(ch, info, "pre-analysis preserve")
                    continue
                if self._stem_ml_corrections_frozen(info):
                    skipped_count += 1
                    self._log_processing_skip(ch, info, "MuQ stem freeze")
                    continue
                preserved_count += 1
                info.was_reset = False

            if self.soundcheck_profile_capture_multiphase_learning and self.audio_capture is not None:
                silence_window = max(0.0, self.soundcheck_profile_silence_capture_duration_sec)
                if silence_window > 0.0:
                    time.sleep(silence_window)
                self._capture_learning_phase(
                    "SILENCE_CAPTURE",
                    RuntimeState.SILENCE_CAPTURE,
                    include_inactive=True,
                    metadata={
                        "reset_count": 0,
                        "preserved_count": preserved_count,
                        "skipped_count": skipped_count,
                    },
                    notes="Pre-analysis noise floor capture with existing mixer processing preserved",
                )

            logger.info(
                "Pre-analysis preserve complete: %s channels preserved, %s skipped for safety",
                preserved_count,
                skipped_count,
            )
            return

        self._set_state(
            EngineState.RESETTING,
            "Resetting channel processing to neutral by explicit operator opt-in..."
        )

        reset_count = 0
        skipped_count = 0

        for ch in self._iter_channels():
            info = self.channels.get(ch)
            if info is None:
                continue
            if not info.auto_corrections_enabled:
                skipped_count += 1
                self._log_processing_skip(ch, info, "reset")
                continue
            if self._stem_ml_corrections_frozen(info):
                skipped_count += 1
                self._log_processing_skip(ch, info, "MuQ stem freeze")
                continue

            try:
                self.mixer_client.reset_channel_processing(ch)
                info.was_reset = True
                reset_count += 1
            except Exception as e:
                logger.warning(f"Ch {ch}: reset failed: {e}")

            # Small delay between channels to avoid overwhelming the mixer
            if reset_count % 8 == 0:
                time.sleep(0.05)

        # Wait for resets to take effect on mixer DSP
        time.sleep(0.5)

        if self.soundcheck_profile_capture_multiphase_learning and self.audio_capture is not None:
            silence_window = max(0.0, self.soundcheck_profile_silence_capture_duration_sec)
            if silence_window > 0.0:
                time.sleep(silence_window)
            self._capture_learning_phase(
                "SILENCE_CAPTURE",
                RuntimeState.SILENCE_CAPTURE,
                include_inactive=True,
                metadata={
                    "reset_count": reset_count,
                    "skipped_count": skipped_count,
                },
                notes="Post-reset noise floor capture before line check",
            )

        logger.info(
            "Reset complete: %s channels set to neutral, %s skipped for safety",
            reset_count,
            skipped_count,
        )

    # ── 4. Wait for signal and analyze ───────────────────────────

    def _wait_and_analyze(self):
        """Wait for audio signals, then run full multi-metric analysis.

        Phase 1 (WAITING): detect signal presence on each channel.
        Phase 2 (ANALYZING): feed 3+ seconds of audio through SignalAnalyzer
        to compute momentary/short-term/integrated LUFS, true peak, RMS,
        crest factor, ADSR envelope, transient metrics, and full spectral
        analysis per channel.
        """
        self._set_state(EngineState.WAITING_FOR_SIGNAL, "Waiting for audio signals...")

        warmup = 2.0
        time.sleep(warmup)

        analysis_start = time.time()
        analysis_timeout = 30.0

        while not self._stop_event.is_set():
            elapsed = time.time() - analysis_start
            if elapsed > analysis_timeout:
                logger.info("Analysis timeout reached, proceeding with available data")
                break

            any_new = False
            for ch, info in self.channels.items():
                if info.has_signal:
                    continue

                peak = self.audio_capture.get_peak(ch, self.block_size * 4)

                if peak > SIGNAL_THRESHOLD_DB:
                    info.has_signal = True
                    info.peak_db = peak
                    any_new = True

                    # Create analyzer for this channel
                    analyzer = SignalAnalyzer(ch, self.sample_rate, self.block_size)
                    self._analyzers[ch] = analyzer
                    info.analyzer = analyzer

                    logger.info(f"Ch {ch} '{info.name}': signal detected (peak={peak:.1f}dB)")

                    if not info.recognized:
                        self._classify_by_spectrum(ch, info)

            channels_with_signal = sum(1 for c in self.channels.values() if c.has_signal)
            if channels_with_signal > 0 and not any_new and elapsed > 10.0:
                logger.info(f"No new signals for a while, proceeding with {channels_with_signal} channels")
                break

            time.sleep(0.5)

        self._capture_learning_phase(
            "LINE_CHECK",
            RuntimeState.LINE_CHECK,
            include_inactive=True,
            metadata={
                "detected_channel_ids": [
                    ch for ch, info in self.channels.items() if info.has_signal
                ],
                "missing_channel_ids": [
                    ch for ch, info in self.channels.items() if not info.has_signal
                ],
                "detected_peaks_db": {
                    str(ch): float(info.peak_db)
                    for ch, info in self.channels.items()
                    if info.has_signal
                },
            },
            notes="Signal detection and channel presence learning",
        )

        # ── Phase 2: Full multi-metric analysis ──────────────────
        self._set_state(EngineState.ANALYZING, "Running full signal analysis...")

        analysis_duration = max(SIGNAL_ANALYSIS_SECONDS, 3.0)
        analysis_blocks = int(analysis_duration * self.sample_rate / self.block_size)
        block_interval = self.block_size / self.sample_rate

        for block_i in range(analysis_blocks):
            if self._stop_event.is_set():
                break

            for ch, info in self.channels.items():
                if not info.has_signal:
                    continue
                analyzer = self._analyzers.get(ch)
                if analyzer is None:
                    continue

                samples = self.audio_capture.get_buffer(ch, self.block_size)
                if len(samples) >= self.block_size:
                    analyzer.process(samples)

            time.sleep(block_interval)

        # ── Collect metrics ──────────────────────────────────────
        for ch, info in self.channels.items():
            if not info.has_signal:
                continue

            analyzer = self._analyzers.get(ch)
            if analyzer:
                m = analyzer.get_metrics()
                info.metrics = m

                info.peak_db = m.level.peak_db
                info.rms_db = m.level.rms_db
                info.lufs = m.level.lufs_integrated if m.level.lufs_integrated > -90 else m.level.lufs_short_term
                info.spectral_centroid = m.spectral.centroid_hz

                logger.info(
                    f"Ch {ch} '{info.name}' metrics: "
                    f"LUFS M={m.level.lufs_momentary:.1f} S={m.level.lufs_short_term:.1f} "
                    f"I={m.level.lufs_integrated:.1f} | "
                    f"TP={m.level.true_peak_dbtp:.1f}dBTP | "
                    f"RMS={m.level.rms_db:.1f} | "
                    f"Crest={m.level.crest_factor_db:.1f} | "
                    f"DR={m.dynamics.dynamic_range_db:.1f}dB | "
                    f"Atk={m.dynamics.attack_time_ms:.0f}ms "
                    f"Dec={m.dynamics.decay_time_ms:.0f}ms "
                    f"Sus={m.dynamics.sustain_level_db:.1f}dB "
                    f"Rel={m.dynamics.release_time_ms:.0f}ms | "
                    f"Trans={m.dynamics.transient_density:.1f}/s "
                    f"Str={m.dynamics.transient_strength_db:.1f}dB | "
                    f"Centroid={m.spectral.centroid_hz:.0f}Hz "
                    f"Bright={m.spectral.brightness:.3f} "
                    f"Mud={m.spectral.mud_ratio:.3f} "
                    f"Pres={m.spectral.presence_ratio:.3f}"
                )

        channel_features, stem_features, contribution_matrix, _ = (
            self._collect_autofoh_monitor_features()
        )
        if channel_features:
            self._capture_learning_phase(
                "SOURCE_LEARNING",
                RuntimeState.SOURCE_LEARNING,
                channel_features=channel_features,
                stem_features=stem_features,
                metadata={
                    "channel_count": len(channel_features),
                },
                notes="Per-channel soundcheck learning after initial analysis pass",
            )
            self._capture_learning_phase(
                "STEM_LEARNING",
                RuntimeState.STEM_LEARNING,
                channel_features=channel_features,
                stem_features=stem_features,
                metadata={
                    "stem_names": sorted(
                        stem_name for stem_name in stem_features.keys()
                        if stem_name != "MASTER"
                    ),
                    "band_contributions": (
                        contribution_matrix.band_contributions if contribution_matrix else {}
                    ),
                },
                notes="Stem aggregation and contribution learning snapshot",
            )
            self._capture_learning_phase(
                "FULL_BAND_LEARNING",
                RuntimeState.FULL_BAND_LEARNING,
                channel_features=channel_features,
                stem_features=stem_features,
                metadata={
                    "learned_target_corridor_preview": (
                        stem_features["MASTER"].slope_compensated_band_levels_db
                        if "MASTER" in stem_features
                        else {}
                    ),
                },
                notes="Full-band learning snapshot before correction application",
            )

    def _classify_by_spectrum(self, ch: int, info: ChannelInfo):
        """Classify an unrecognized channel by spectral characteristics."""
        spec = self.audio_capture.get_spectrum(ch, 4096)
        freqs = spec["frequencies"]
        mags_db = spec["magnitude_db"]

        if len(freqs) == 0:
            return

        linear = 10 ** (mags_db / 20.0)
        total = np.sum(linear) + 1e-10
        centroid = float(np.sum(freqs * linear) / total)

        low_mask = (freqs >= 100) & (freqs < 300)
        mid_mask = (freqs >= 1000) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 10000)

        energy_bands = {
            "low_100_300": float(np.mean(linear[low_mask])) if np.any(low_mask) else 0.0,
            "mid_1k_4k": float(np.mean(linear[mid_mask])) if np.any(mid_mask) else 0.0,
            "high_4k_10k": float(np.mean(linear[high_mask])) if np.any(high_mask) else 0.0,
        }
        max_e = max(energy_bands.values()) + 1e-10
        energy_bands = {k: v / max_e for k, v in energy_bands.items()}

        preset = recognize_instrument_spectral_fallback(
            info.name, centroid, energy_bands
        )
        if preset:
            classification = classification_from_legacy_preset(
                preset,
                channel_name=info.name,
                confidence=0.45,
                match_type="spectral_fallback",
            )
            self._apply_classification_to_info(info, classification.to_dict())
            logger.info(
                "Ch %s '%s': spectral classification -> %s (centroid=%.0fHz, auto_apply=%s)",
                ch,
                info.name,
                preset,
                centroid,
                info.auto_corrections_enabled,
            )

    def _select_lead_channels(self) -> List[int]:
        lead_channels = []
        for ch, info in self.channels.items():
            if not info.has_signal:
                continue
            if info.source_role == "lead_vocal" or "LEAD" in info.stem_roles:
                lead_channels.append(ch)
        return lead_channels

    def _channel_metadata_for_profile(self, include_inactive: bool = False) -> Dict[int, Dict[str, Any]]:
        return {
            ch: {
                "name": info.name,
                "source_role": info.source_role,
                "stem_roles": list(info.stem_roles),
                "allowed_controls": list(info.allowed_controls),
                "priority": float(info.priority),
            }
            for ch, info in self.channels.items()
            if include_inactive or info.has_signal
        }

    def _collect_autofoh_features(
        self,
        include_inactive: bool = False,
        runtime_state: Optional[RuntimeState] = None,
    ):
        if self.audio_capture is None:
            return {}, {}, None, {}

        fft_size = int(self.autofoh_analysis_config.get("fft_size", 4096))
        octave_fraction = int(self.autofoh_analysis_config.get("octave_fraction", 3))
        slope_db = float(
            self.autofoh_analysis_config.get("slope_compensation_db_per_octave", 4.5)
        )
        target_corridor = self._target_corridor_for_runtime_state(runtime_state)
        channel_features = {}
        channel_stems = {}

        for ch, info in self.channels.items():
            if not include_inactive and not info.has_signal:
                continue
            samples = self.audio_capture.get_buffer(ch, fft_size)
            if len(samples) < max(fft_size // 2, self.block_size):
                continue
            channel_features[ch] = extract_analysis_features(
                samples,
                sample_rate=self.sample_rate,
                fft_size=fft_size,
                octave_fraction=octave_fraction,
                slope_compensation_db_per_octave=slope_db,
                target_corridor=target_corridor,
            )
            stems = [stem for stem in info.stem_roles if stem and stem != "MASTER"]
            if not stems:
                stems = ["LEAD"] if info.source_role == "lead_vocal" else ["UNKNOWN"]
            channel_stems[ch] = stems

        if not channel_features:
            return {}, {}, None, {}

        stem_features = aggregate_stem_features(
            channel_features,
            channel_stems,
            target_corridor=target_corridor,
        )
        contribution_inputs = {
            stem_name: feature
            for stem_name, feature in stem_features.items()
            if stem_name != "MASTER"
        }
        contribution_matrix = build_stem_contribution_matrix(contribution_inputs)
        return channel_features, stem_features, contribution_matrix, channel_stems

    def _collect_autofoh_monitor_features(
        self,
        runtime_state: Optional[RuntimeState] = None,
    ):
        return self._collect_autofoh_features(
            include_inactive=False,
            runtime_state=runtime_state or self.runtime_state,
        )

    def _read_live_settings_for_channel(
        self,
        ch: int,
        info: ChannelInfo,
    ) -> Dict[str, Any]:
        """Read the current console state before a shared live-mix decision."""
        raw_settings = self._snapshot_raw(info)
        if self.mixer_client is None or not hasattr(self.mixer_client, "get_channel_settings"):
            return raw_settings
        try:
            current_settings = self.mixer_client.get_channel_settings(ch) or {}
        except Exception as exc:
            logger.debug("Ch %s: live shared-mix state read failed: %s", ch, exc)
            return raw_settings

        merged = dict(raw_settings)
        merged.update(current_settings)
        info.fader_db = self._as_float(merged.get("fader_db"), info.fader_db)
        return merged

    def _current_eq_gain_map_from_settings(self, raw_settings: Dict[str, Any]) -> Dict[int, float]:
        gains: Dict[int, float] = {}
        for idx, band in enumerate((raw_settings.get("eq_bands") or [])[:4], start=1):
            if isinstance(band, (list, tuple)) and len(band) >= 2:
                gains[idx] = self._as_float(band[1], 0.0)
        return gains

    def _build_live_shared_mix_channels(self) -> List[LiveSharedMixChannel]:
        if self.audio_capture is None:
            return []

        window_samples = max(
            self.block_size,
            int(max(1.0, self.shared_chat_mix_config.analysis_window_sec) * self.sample_rate),
        )
        channels: List[LiveSharedMixChannel] = []
        for ch in self._iter_channels():
            info = self.channels.get(ch)
            if info is None or not info.has_signal:
                continue
            if ch in self.master_reference_channels or "MASTER" in info.stem_roles:
                continue

            raw_settings = self._read_live_settings_for_channel(ch, info)
            try:
                audio = np.asarray(
                    self.audio_capture.get_buffer(ch, window_samples),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception as exc:
                logger.debug("Ch %s: live shared-mix audio capture failed: %s", ch, exc)
                continue
            if audio.size < max(1024, self.block_size // 2):
                continue

            role = normalize_live_role(
                preset=info.preset or "",
                source_role=info.source_role or "",
                name=info.name or "",
            )
            stems = tuple(stem for stem in info.stem_roles if stem and stem != "MASTER")
            if not stems:
                stems = (role.upper(),) if role != "unknown" else ("UNKNOWN",)

            channels.append(
                LiveSharedMixChannel(
                    channel_id=ch,
                    name=info.name or f"Ch {ch}",
                    role=role,
                    stems=stems,
                    priority=float(info.priority),
                    audio=audio.copy(),
                    sample_rate=self.sample_rate,
                    fader_db=self._as_float(raw_settings.get("fader_db"), info.fader_db),
                    muted=self._as_bool(raw_settings.get("muted"), False),
                    auto_corrections_enabled=bool(info.auto_corrections_enabled),
                    raw_settings=raw_settings,
                    current_eq_gain=self._current_eq_gain_map_from_settings(raw_settings),
                    current_hpf_hz=self._as_float(raw_settings.get("hpf_freq"), 20.0),
                    hpf_enabled=self._as_bool(raw_settings.get("hpf_enabled"), False),
                )
            )
        return channels

    def _capture_master_reference_audio(self, window_sec: Optional[float] = None) -> Optional[np.ndarray]:
        if self.audio_capture is None or not self.master_reference_channels:
            return None

        analysis_window_sec = (
            self.shared_chat_mix_config.analysis_window_sec
            if window_sec is None
            else float(window_sec)
        )
        window_samples = max(
            self.block_size,
            int(max(1.0, analysis_window_sec) * self.sample_rate),
        )
        buffers: List[np.ndarray] = []
        for ch in sorted(self.master_reference_channels)[:2]:
            if ch > self.num_channels:
                continue
            try:
                samples = np.asarray(
                    self.audio_capture.get_buffer(ch, window_samples),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception as exc:
                logger.debug("Master reference Ch %s capture failed: %s", ch, exc)
                continue
            if samples.size >= max(1024, self.block_size // 2):
                buffers.append(samples.copy())

        if not buffers:
            return None
        min_len = min(buffer.size for buffer in buffers)
        if min_len <= 0:
            return None
        aligned = [buffer[-min_len:] for buffer in buffers]
        if len(aligned) == 1:
            return aligned[0]
        return np.column_stack(aligned[:2]).astype(np.float32)

    def _read_main_fader_for_live_mix(self) -> Optional[float]:
        if self.mixer_client is None or not hasattr(self.mixer_client, "get_main_fader"):
            return None
        try:
            value = self.mixer_client.get_main_fader(1)
        except Exception as exc:
            logger.debug("Main 1 fader readback failed before shared mix: %s", exc)
            return None
        if value is None:
            return None
        return self._as_float(value, 0.0)

    def _apply_live_shared_mix_pass(
        self,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ) -> List[SafetyDecision]:
        if not self.shared_chat_mix_config.enabled:
            return []
        if self.audio_capture is None or self.safety_controller is None:
            return []

        channels = self._build_live_shared_mix_channels()
        if not channels:
            self._log_autofoh_event(
                "live_shared_chat_mix_plan",
                enabled=True,
                reason="no_eligible_source_channels",
            )
            return []

        master_audio = self._capture_master_reference_audio()
        main_fader = self._read_main_fader_for_live_mix()
        plan = build_live_shared_mix_plan(
            channels,
            self.sample_rate,
            config=self.shared_chat_mix_config,
            master_audio=master_audio,
            master_current_fader_db=main_fader,
        )
        self._log_autofoh_event(
            "live_shared_chat_mix_plan",
            report=plan.report,
        )
        self._emit_observation(
            message=(
                "[AUTOFOH] Live shared-mix analysis: "
                f"{plan.report.get('actions_planned', 0)} action(s), "
                f"master={plan.report.get('master', {}).get('action', 'n/a')}"
            ),
            summary={
                "live_shared_chat_mix": plan.report,
            },
        )

        decisions: List[SafetyDecision] = []
        for action in plan.actions:
            action_channel_id = getattr(action, "channel_id", None)
            target_info = self.channels.get(action_channel_id) if action_channel_id is not None else None
            control_name = self._typed_action_control_name(action)
            if (
                control_name is not None
                and target_info is not None
                and not self._control_allowed(target_info, control_name)
            ):
                self._log_processing_skip(
                    action_channel_id,
                    target_info,
                    "live shared-mix correction",
                )
                continue

            requested_action = action
            decision = self._execute_action(
                action,
                evaluation_context={
                    "problem_type": "live_shared_chat_mix",
                    "expected_effect": getattr(action, "reason", ""),
                    "source_rules": list(plan.report.get("rules", [])),
                    "analysis_before": (
                        plan.report.get("analysis_before", {})
                        .get("band_deviation_db", {})
                    ),
                    "planned_action_types": list(plan.report.get("planned_action_types", [])),
                },
                phase_guard_context=phase_guard_context,
            )
            if decision is None:
                continue
            decisions.append(decision)
            logger.info(
                "Live shared-mix decision: %s ch=%s sent=%s allowed=%s rate_limited=%s msg=%s reason=%s",
                decision.action.action_type,
                getattr(decision.action, "channel_id", getattr(decision.action, "main_id", None)),
                decision.sent,
                decision.allowed,
                decision.rate_limited,
                decision.message,
                getattr(requested_action, "reason", ""),
            )
            if decision.sent and isinstance(decision.action, ChannelFaderMove):
                info = self.channels.get(decision.action.channel_id)
                if info is not None:
                    info.fader_db = decision.action.target_db

        sent_count = sum(1 for decision in decisions if decision.sent)
        decision_payload = [
            {
                "action_type": decision.action.action_type,
                "channel": getattr(decision.action, "channel_id", None),
                "main": getattr(decision.action, "main_id", None),
                "band": getattr(decision.action, "band", None),
                "freq_hz": round(float(getattr(decision.action, "freq_hz", 0.0)), 1)
                if hasattr(decision.action, "freq_hz")
                else None,
                "gain_db": round(float(getattr(decision.action, "gain_db", 0.0)), 2)
                if hasattr(decision.action, "gain_db")
                else None,
                "sent": bool(decision.sent),
                "allowed": bool(decision.allowed),
                "bounded": bool(decision.bounded),
                "rate_limited": bool(decision.rate_limited),
                "message": decision.message,
                "reason": getattr(decision.action, "reason", ""),
            }
            for decision in decisions
        ]
        self._emit_observation(
            message=(
                "[AUTOFOH] Live shared-mix applied: "
                f"{sent_count}/{len(plan.actions)} action(s) sent"
            ),
            summary={
                "live_shared_chat_mix_applied": {
                    "planned": len(plan.actions),
                    "sent": sent_count,
                    "decisions": decision_payload,
                }
            },
        )
        logger.info(
            "Live shared-mix pass complete: planned=%s sent=%s",
            len(plan.actions),
            sent_count,
        )
        return decisions

    def _apply_group_bus_corrections(
        self,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ) -> List[SafetyDecision]:
        """Apply bounded level corrections to routed group buses, excluding monitors."""
        if not self.group_bus_correction_enabled:
            return []
        if self.safety_controller is None or self.mixer_client is None:
            return []

        self._read_group_state_for_channels()
        decisions: List[SafetyDecision] = []
        skipped_monitor = []
        planned = []

        for bus_id, snapshot in sorted(self.bus_snapshots.items()):
            if self._is_monitor_bus(bus_id):
                skipped_monitor.append(bus_id)
                continue
            if snapshot.muted:
                continue

            source_infos = [
                self.channels[ch]
                for ch in snapshot.source_channels
                if ch in self.channels
                and self.channels[ch].has_signal
                and self.channels[ch].auto_corrections_enabled
                and not self._stem_ml_corrections_frozen(self.channels[ch])
            ]
            if not source_infos:
                continue

            current_fader = self._safe_call(
                lambda bus=bus_id: self.mixer_client.get_bus_fader(bus),
                snapshot.fader_db,
            )
            current_fader = self._as_float(current_fader, snapshot.fader_db)
            target_fader = current_fader
            reasons = []

            if current_fader > 0.0:
                target_fader = 0.0
                reasons.append("bus fader above 0dB ceiling")

            hot_peaks = [
                (info.metrics.level.true_peak_dbtp if info.metrics else info.peak_db)
                for info in source_infos
                if (info.metrics and info.metrics.level.true_peak_dbtp > -90.0) or info.peak_db > -90.0
            ]
            if hot_peaks and max(hot_peaks) > -4.0:
                target_fader = min(target_fader, current_fader - 0.5)
                reasons.append("routed source true peak close to headroom ceiling")

            loud_deviations = []
            for info in source_infos:
                measured_lufs = info.lufs
                if info.metrics and info.metrics.level.lufs_integrated > -90.0:
                    measured_lufs = info.metrics.level.lufs_integrated
                target_lufs = INSTRUMENT_TARGET_LUFS.get(info.preset or "custom", -23.0)
                if measured_lufs > -90.0:
                    loud_deviations.append(measured_lufs - target_lufs)
            if loud_deviations and float(np.mean(loud_deviations)) > 2.5:
                target_fader = min(target_fader, current_fader - 0.5)
                reasons.append("routed source group is above learned balance target")

            if target_fader >= current_fader - 0.25:
                continue

            target_fader = max(-144.0, min(0.0, target_fader))
            decision = self._execute_action(
                BusFaderMove(
                    bus_id=bus_id,
                    target_db=round(target_fader, 2),
                    reason=(
                        f"Group BUS correction for {snapshot.name or bus_id}: "
                        + "; ".join(reasons)
                    ),
                ),
                register_evaluation=False,
                evaluation_context={
                    "problem_type": "group_bus_balance",
                    "expected_effect": "Reduce routed bus level while preserving monitor buses",
                    "source_channels": list(snapshot.source_channels),
                    "monitor_bus_ids": sorted(self.monitor_bus_ids),
                },
                phase_guard_context=phase_guard_context,
            )
            if decision is not None:
                decisions.append(decision)
                planned.append(
                    {
                        "bus": bus_id,
                        "name": snapshot.name,
                        "current_fader_db": round(float(current_fader), 2),
                        "requested_target_db": round(float(target_fader), 2),
                        "sent": bool(decision.sent),
                        "message": decision.message,
                    }
                )
                if decision.sent and isinstance(decision.action, BusFaderMove):
                    snapshot.fader_db = decision.action.target_db

        if skipped_monitor or planned:
            self._emit_observation(
                message=(
                    "[AUTOFOH] Group bus correction: "
                    f"{len(planned)} action(s), monitor buses skipped={skipped_monitor}"
                ),
                summary={
                    "group_bus_corrections": planned,
                    "monitor_bus_ids": sorted(self.monitor_bus_ids),
                    "skipped_monitor_buses": skipped_monitor,
                },
            )
        return decisions

    def _capture_learning_phase(
        self,
        phase_name: str,
        runtime_state: RuntimeState,
        *,
        include_inactive: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        notes: str = "",
        channel_features: Optional[Dict[int, Any]] = None,
        stem_features: Optional[Dict[str, Any]] = None,
    ) -> Optional[PhaseLearningSnapshot]:
        if not self.soundcheck_profile_capture_multiphase_learning:
            return None

        if channel_features is None or stem_features is None:
            channel_features, stem_features, _, _ = self._collect_autofoh_features(
                include_inactive=include_inactive,
                runtime_state=runtime_state,
            )
        if not channel_features and not stem_features and not metadata and not notes:
            return None

        snapshot = build_phase_learning_snapshot(
            phase_name=phase_name,
            runtime_state=runtime_state.value,
            channel_features=channel_features,
            stem_features=stem_features,
            active_channel_ids=sorted(channel_features.keys()),
            metadata={
                "engine_state": self.state.value,
                "selected_channels": self._iter_channels(),
                **dict(metadata or {}),
            },
            notes=notes,
        )
        self._phase_learning_snapshots[phase_name] = snapshot
        self._log_autofoh_event(
            "soundcheck_learning_phase_captured",
            phase_name=phase_name,
            runtime_state=runtime_state.value,
            active_channel_ids=list(snapshot.active_channel_ids),
            notes=notes,
            metadata=snapshot.metadata,
            master_features=self._feature_snapshot_payload(snapshot.master_features),
        )
        return snapshot

    def _build_soundcheck_profile_from_live_buffers(self) -> Optional[AutoFOHSoundcheckProfile]:
        channel_features, stem_features, contribution_matrix, channel_stems = (
            self._collect_autofoh_monitor_features(runtime_state=self.runtime_state)
        )
        if not channel_features:
            return None

        profile = build_soundcheck_profile(
            channel_features=channel_features,
            channel_metadata=self._channel_metadata_for_profile(),
            stem_features=stem_features,
            stem_contributions=contribution_matrix.band_contributions if contribution_matrix else {},
            sample_rate=self.sample_rate,
            profile_name=f"autofoh_soundcheck_{int(time.time())}",
            phase_snapshots=self._phase_learning_snapshots,
            metadata={
                "runtime_state": self.runtime_state.value,
                "engine_state": self.state.value,
                "selected_channels": self._iter_channels(),
                "mixer_type": self.mixer_type,
            },
        )
        return profile

    @staticmethod
    def _typed_action_control_name(action) -> Optional[str]:
        if isinstance(action, ChannelGainMove):
            return "gain"
        if isinstance(action, ChannelEQMove):
            return "eq"
        if isinstance(action, HighPassAdjust):
            return "hpf"
        if isinstance(action, CompressorAdjust):
            return "compressor"
        if isinstance(action, CompressorMakeupAdjust):
            return "compressor"
        if isinstance(action, GateAdjust):
            return "compressor"
        if isinstance(action, PanAdjust):
            return "pan"
        if isinstance(action, SendLevelAdjust):
            return "fx_send"
        if isinstance(action, ChannelFaderMove):
            return "fader"
        return None

    def _phase_target_action_guard(
        self,
        action,
        *,
        channel_features: Dict[int, Any],
        stem_features: Dict[str, Any],
        channel_stems: Dict[int, Any],
        runtime_state: RuntimeState,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if not self.soundcheck_profile_use_phase_target_action_guards:
            return True, "", {}

        phase_target = self._phase_target_for_runtime_state(runtime_state)
        if phase_target is None:
            return True, "", {}

        channel_id = getattr(action, "channel_id", None)
        if channel_id is None:
            return True, "", {}

        features = channel_features.get(channel_id)
        if features is None:
            return True, "", {}

        info = self.channels.get(channel_id)
        metadata = {
            "phase_name": phase_target.phase_name,
            "phase_runtime_state": phase_target.runtime_state,
            "channel_id": int(channel_id),
            "action_type": action.action_type,
        }

        if isinstance(action, ChannelFaderMove):
            source_role = info.source_role if info is not None else "unknown"
            expected_rms = phase_target.expected_channel_rms_db.get(channel_id)
            if expected_rms is None:
                expected_rms = phase_target.expected_source_role_rms_db.get(source_role)
            if expected_rms is None:
                return True, "", metadata

            current_rms = float(features.rms_db)
            drift_db = current_rms - float(expected_rms)
            direction_db = float(action.delta_db)
            if abs(direction_db) < 1e-6 and info is not None:
                direction_db = float(action.target_db) - float(info.fader_db)

            metadata.update(
                {
                    "expected_channel_rms_db": float(expected_rms),
                    "current_channel_rms_db": current_rms,
                    "channel_rms_drift_db": drift_db,
                    "channel_level_tolerance_db": float(
                        phase_target.channel_level_tolerance_db
                    ),
                    "requested_fader_delta_db": direction_db,
                }
            )
            tolerance_db = float(phase_target.channel_level_tolerance_db)
            if direction_db > 0.0 and drift_db > tolerance_db:
                return (
                    False,
                    "phase target guard blocked boost; channel input already above learned baseline",
                    metadata,
                )
            if direction_db < 0.0 and drift_db < -tolerance_db:
                return (
                    False,
                    "phase target guard blocked cut; channel input already below learned baseline",
                    metadata,
                )
            return True, "", metadata

        if isinstance(action, ChannelGainMove):
            source_role = info.source_role if info is not None else "unknown"
            expected_rms = phase_target.expected_channel_rms_db.get(channel_id)
            if expected_rms is None:
                expected_rms = phase_target.expected_source_role_rms_db.get(source_role)
            if expected_rms is None:
                return True, "", metadata

            current_rms = float(features.rms_db)
            drift_db = current_rms - float(expected_rms)
            requested_gain_db = float(action.target_db)
            tolerance_db = float(phase_target.channel_level_tolerance_db)
            metadata.update(
                {
                    "expected_channel_rms_db": float(expected_rms),
                    "current_channel_rms_db": current_rms,
                    "channel_rms_drift_db": drift_db,
                    "channel_level_tolerance_db": tolerance_db,
                    "requested_gain_db": requested_gain_db,
                }
            )
            if requested_gain_db > 0.0 and drift_db > tolerance_db:
                return (
                    False,
                    "phase target guard blocked input gain boost; channel input already above learned baseline",
                    metadata,
                )
            if requested_gain_db < 0.0 and drift_db < -tolerance_db:
                return (
                    False,
                    "phase target guard blocked input gain cut; channel input already below learned baseline",
                    metadata,
                )
            return True, "", metadata

        if isinstance(action, ChannelEQMove):
            is_mirror_eq = str(getattr(action, "reason", "")).startswith("Mirror EQ")
            band_name = infer_evaluation_band(action)
            if band_name:
                current_band_level = float(
                    features.slope_compensated_band_levels_db.get(band_name, -100.0)
                )
                target_band_level = float(
                    phase_target.target_corridor.target_for_band(band_name)
                )
                band_error_db = current_band_level - target_band_level
                metadata.update(
                    {
                        "band_name": band_name,
                        "current_band_level_db": current_band_level,
                        "target_band_level_db": target_band_level,
                        "band_error_db": band_error_db,
                        "green_delta_db": float(
                            phase_target.target_corridor.green_delta_db
                        ),
                        "requested_eq_gain_db": float(action.gain_db),
                    }
                )
                green_delta_db = float(phase_target.target_corridor.green_delta_db)
                if is_mirror_eq:
                    metadata["green_corridor_decision"] = (
                        "allow_mirror_eq_source_separation"
                    )
                elif action.gain_db < 0.0 and band_error_db <= green_delta_db:
                    return (
                        False,
                        "phase target guard blocked EQ cut inside learned green corridor",
                        metadata,
                    )
                if action.gain_db > 0.0 and band_error_db >= -green_delta_db:
                    return (
                        False,
                        "phase target guard blocked EQ boost inside learned green corridor",
                        metadata,
                    )

            stem_names = [
                stem_name
                for stem_name in (channel_stems.get(channel_id) or getattr(info, "stem_roles", []))
                if stem_name and stem_name != "MASTER"
            ]
            tolerance_db = float(phase_target.stem_level_tolerance_db)
            for stem_name in stem_names:
                expected_stem_rms = phase_target.expected_stem_rms_db.get(stem_name)
                current_stem_features = stem_features.get(stem_name)
                if expected_stem_rms is None or current_stem_features is None:
                    continue
                stem_drift_db = float(current_stem_features.rms_db) - float(expected_stem_rms)
                if action.gain_db < 0.0 and stem_drift_db < -tolerance_db:
                    metadata.update(
                        {
                            "guard_stem": stem_name,
                            "expected_stem_rms_db": float(expected_stem_rms),
                            "current_stem_rms_db": float(current_stem_features.rms_db),
                            "stem_rms_drift_db": stem_drift_db,
                            "stem_level_tolerance_db": tolerance_db,
                        }
                    )
                    return (
                        False,
                        f"phase target guard blocked EQ cut; stem {stem_name} is already below learned baseline",
                        metadata,
                    )
                if action.gain_db > 0.0 and stem_drift_db > tolerance_db:
                    metadata.update(
                        {
                            "guard_stem": stem_name,
                            "expected_stem_rms_db": float(expected_stem_rms),
                            "current_stem_rms_db": float(current_stem_features.rms_db),
                            "stem_rms_drift_db": stem_drift_db,
                            "stem_level_tolerance_db": tolerance_db,
                        }
                    )
                    return (
                        False,
                        f"phase target guard blocked EQ boost; stem {stem_name} is already above learned baseline",
                        metadata,
                    )

        if isinstance(action, HighPassAdjust):
            bounds = phase_target.hpf_frequency_range_hz_by_channel.get(channel_id)
            if not bounds:
                return True, "", metadata
            min_hz = float(bounds.get("min_hz", 20.0))
            max_hz = float(bounds.get("max_hz", 400.0))
            requested_hz = float(action.freq_hz)
            metadata.update(
                {
                    "requested_hpf_hz": requested_hz,
                    "hpf_min_hz": min_hz,
                    "hpf_max_hz": max_hz,
                }
            )
            if requested_hz < min_hz:
                return (
                    False,
                    "phase target guard blocked HPF below learned low-end protection floor",
                    metadata,
                )
            if requested_hz > max_hz:
                return (
                    False,
                    "phase target guard blocked HPF above learned tonal-preservation ceiling",
                    metadata,
                )
            return True, "", metadata

        if isinstance(action, CompressorAdjust):
            threshold_bounds = phase_target.compressor_threshold_range_db_by_channel.get(
                channel_id
            )
            ratio_bounds = phase_target.compressor_ratio_range_by_channel.get(channel_id)
            if not threshold_bounds and not ratio_bounds:
                return True, "", metadata
            requested_threshold_db = float(action.threshold_db)
            requested_ratio = float(action.ratio)
            metadata.update(
                {
                    "requested_compressor_threshold_db": requested_threshold_db,
                    "requested_compressor_ratio": requested_ratio,
                }
            )
            if threshold_bounds:
                min_db = float(threshold_bounds.get("min_db", -50.0))
                max_db = float(threshold_bounds.get("max_db", -5.0))
                metadata.update(
                    {
                        "compressor_threshold_min_db": min_db,
                        "compressor_threshold_max_db": max_db,
                    }
                )
                if requested_threshold_db < min_db:
                    return (
                        False,
                        "phase target guard blocked compressor threshold below learned range",
                        metadata,
                    )
                if requested_threshold_db > max_db:
                    return (
                        False,
                        "phase target guard blocked compressor threshold above learned range",
                        metadata,
                    )
            if ratio_bounds:
                min_ratio = float(ratio_bounds.get("min_ratio", 1.0))
                max_ratio = float(ratio_bounds.get("max_ratio", 10.0))
                metadata.update(
                    {
                        "compressor_ratio_min": min_ratio,
                        "compressor_ratio_max": max_ratio,
                    }
                )
                if requested_ratio < min_ratio:
                    return (
                        False,
                        "phase target guard blocked compressor ratio below learned range",
                        metadata,
                    )
                if requested_ratio > max_ratio:
                    return (
                        False,
                        "phase target guard blocked compressor ratio above learned range",
                        metadata,
                    )
            return True, "", metadata

        if isinstance(action, SendLevelAdjust):
            bounds = phase_target.fx_send_level_range_db_by_channel.get(channel_id)
            if not bounds:
                return True, "", metadata
            min_db = float(bounds.get("min_db", -40.0))
            max_db = float(bounds.get("max_db", -5.0))
            requested_db = float(action.level_db)
            metadata.update(
                {
                    "send_bus": int(action.send_bus),
                    "requested_fx_send_db": requested_db,
                    "fx_send_min_db": min_db,
                    "fx_send_max_db": max_db,
                }
            )
            if requested_db < min_db:
                return (
                    False,
                    "phase target guard blocked FX send below learned ambience floor",
                    metadata,
                )
            if requested_db > max_db:
                return (
                    False,
                    "phase target guard blocked FX send above learned ambience ceiling",
                    metadata,
                )
            return True, "", metadata

        return True, "", metadata

    def _run_autofoh_monitor_analysis(self):
        if (
            not self.auto_apply
            or self.audio_capture is None
            or self.safety_controller is None
        ):
            return []

        now = time.monotonic()
        if (now - self._last_autofoh_monitor_analysis_at) < self.monitor_cycle_interval_sec:
            return []
        self._last_autofoh_monitor_analysis_at = now

        channel_features, stem_features, contribution_matrix, channel_stems = (
            self._collect_autofoh_monitor_features()
        )
        if not channel_features or contribution_matrix is None:
            return []
        phase_guard_context = {
            "channel_features": channel_features,
            "stem_features": stem_features,
            "channel_stems": channel_stems,
        }

        current_faders = {
            ch: info.fader_db
            for ch, info in self.channels.items()
            if info.has_signal
        }
        channel_priorities = {
            ch: info.priority
            for ch, info in self.channels.items()
            if info.has_signal
        }
        recommendations = []
        lead_channels = self._select_lead_channels()

        if self.lead_masking_analyzer and lead_channels:
            lead_result = self.lead_masking_analyzer.analyze(
                channel_features=channel_features,
                channel_stems=channel_stems,
                stem_features=stem_features,
                contribution_matrix=contribution_matrix,
                lead_channel_ids=lead_channels,
                current_faders_db=current_faders,
                lead_priorities=channel_priorities,
                runtime_state=self.runtime_state,
            )
            if lead_result.problem:
                recommendations.append(
                    ("lead_masking", lead_result.problem, lead_result.candidate_actions)
                )

        master_features = stem_features.get("MASTER")
        if master_features is not None:
            if self.low_end_analyzer is not None:
                low_end_result = self.low_end_analyzer.analyze(
                    master_features=master_features,
                    contribution_matrix=contribution_matrix,
                    channel_features=channel_features,
                    channel_stems=channel_stems,
                )
                if low_end_result.problem:
                    recommendations.append(
                        ("low_end", low_end_result.problem, low_end_result.candidate_actions)
                    )

            for label, detector in (
                ("mud_excess", self.mud_detector),
                ("harshness_excess", self.harshness_detector),
                ("sibilance_excess", self.sibilance_detector),
            ):
                if detector is None:
                    continue
                recommendation = detector.observe(
                    master_features=master_features,
                    contribution_matrix=contribution_matrix,
                    channel_features=channel_features,
                    channel_stems=channel_stems,
                )
                if recommendation.problem:
                    recommendations.append(
                        (label, recommendation.problem, recommendation.candidate_actions)
                    )

        if not recommendations:
            return []

        applied_decisions = []
        label, problem, actions = recommendations[0]
        if problem is not None:
            logger.info(
                "AutoFOH detector '%s': culprit=%s band=%s persistence=%.0f",
                label,
                problem.stem or problem.channel_id,
                problem.band_name,
                problem.persistence_sec,
            )

        for action in actions[:1]:
            control_name = self._typed_action_control_name(action)
            action_channel_id = getattr(action, "channel_id", None)
            target_info = self.channels.get(action_channel_id)
            if (
                control_name is not None
                and target_info is not None
                and not self._control_allowed(target_info, control_name)
            ):
                self._log_processing_skip(
                    getattr(action, "channel_id", "?"),
                    target_info,
                    f"detector-driven {label}",
                )
                continue
            decision = self._execute_action(
                action,
                evaluation_context={
                    "problem_type": label,
                    "band_name": problem.band_name if problem else None,
                    "stem": problem.stem if problem else None,
                    "expected_effect": problem.expected_effect if problem else "",
                    "problem_confidence": (
                        problem.confidence_risk.problem_confidence if problem else 0.0
                    ),
                    "culprit_confidence": (
                        problem.confidence_risk.culprit_confidence if problem else 0.0
                    ),
                    "action_confidence": (
                        problem.confidence_risk.action_confidence if problem else 0.0
                    ),
                    "risk_score": (
                        problem.confidence_risk.risk_score if problem else 1.0
                    ),
                    "pre_features": channel_features.get(action_channel_id),
                },
                phase_guard_context=phase_guard_context,
            )
            if decision is None:
                continue
            if decision.sent and isinstance(decision.action, ChannelFaderMove):
                self.channels[decision.action.channel_id].fader_db = decision.action.target_db
            if decision.sent:
                applied_decisions.append(decision)
                logger.info(
                    "AutoFOH applied '%s': %s -> %s",
                    label,
                    action.action_type,
                    decision.message,
                )
                break
            logger.info(
                "AutoFOH blocked '%s': %s",
                label,
                decision.message,
            )
        return applied_decisions

    # ── 5. Apply corrections ─────────────────────────────────────

    def _apply_corrections(self):
        """Apply full channel strip processing to all active channels.

        Order: input trim → polarity/delay → HPF/LPF → expander → EQ →
        compressor with makeup → pan → fader → FX sends.
        Then: phase/polarity check across correlated pairs.

        Existing console state is preserved by default. Targets are interpreted
        as bounded corrections relative to the snapshot captured before analysis.
        """
        self._set_state(EngineState.APPLYING, "Applying bounded corrections over current mixer state...")
        phase_guard_context = self._build_phase_target_guard_context(
            runtime_state=self.runtime_state
        )

        for ch, info in self.channels.items():
            if not info.has_signal:
                continue
            if ch in self._applied_channels:
                continue
            if not info.auto_corrections_enabled:
                self._log_processing_skip(ch, info, "auto-processing")
                continue
            if self._stem_ml_corrections_frozen(info):
                self._log_processing_skip(ch, info, "MuQ stem freeze")
                continue

            preset = info.preset or "custom"
            had_proc = ""
            if info.original_snapshot and info.original_snapshot.had_processing:
                had_proc = " (existing settings preserved)"
            logger.info(
                f"Ch {ch} '{info.name}' (preset={preset}): "
                f"applying bounded correction chain{had_proc}..."
            )

            try:
                # 1. Input gain staging (preamp trim)
                if self._control_allowed(info, "gain"):
                    self._apply_input_gain(
                        ch,
                        info,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )
                # 2. HPF
                if self._control_allowed(info, "hpf"):
                    self._apply_hpf(
                        ch,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )
                # 2b. Gate/expander for close drum mics so bleed is not what drives EQ/trim.
                self._apply_gate(
                    ch,
                    info,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
                # 3. Parametric EQ (4-band)
                if self._control_allowed(info, "eq"):
                    self._apply_eq(
                        ch,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )
                # 4. Compressor
                if self._control_allowed(info, "compressor"):
                    self._apply_compressor(
                        ch,
                        info,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )
                # 5. Pan
                if self._control_allowed(info, "pan"):
                    self._apply_pan(
                        ch,
                        info,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )
                # 6. Gain correction (output-side, LUFS-based)
                if self._control_allowed(info, "fader"):
                    self._apply_gain_correction(ch, info, preset)
                # 7. Fader
                if self._control_allowed(info, "fader"):
                    self._apply_fader(
                        ch,
                        info,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )
                # 8. FX sends (reverb/delay)
                if self._control_allowed(info, "fx_send"):
                    self._apply_fx_sends(
                        ch,
                        info,
                        preset,
                        phase_guard_context=phase_guard_context,
                    )

                self._applied_channels.add(ch)

                if self.on_channel_update:
                    self.on_channel_update(ch, {
                        "name": info.name,
                        "preset": preset,
                        "source_role": info.source_role,
                        "stem_roles": info.stem_roles,
                        "classification_confidence": info.classification_confidence,
                        "auto_corrections_enabled": info.auto_corrections_enabled,
                        "peak_db": info.peak_db,
                        "rms_db": info.rms_db,
                        "lufs": info.lufs,
                        "gain_correction_db": info.gain_correction_db,
                        "eq_applied": info.eq_applied,
                        "compressor_applied": info.compressor_applied,
                        "fader_db": info.fader_db,
                    })

            except Exception as e:
                logger.error(f"Ch {ch}: error applying corrections: {e}")

        # 9. Full-band shared-chat mix pass over the current live console state.
        self._apply_live_shared_mix_pass(phase_guard_context=phase_guard_context)

        # 10. Group bus balance/headroom pass; BUS 13-16 monitor buses are excluded by policy.
        self._apply_group_bus_corrections(phase_guard_context=phase_guard_context)

        # 11. Phase / polarity check across correlated channel pairs
        self._detect_and_fix_phase()
        snapshot_profile = self._build_soundcheck_profile_from_live_buffers()
        if snapshot_profile is not None:
            self._capture_learning_phase(
                "SNAPSHOT_LOCK",
                RuntimeState.SNAPSHOT_LOCK,
                metadata={
                    "applied_channel_ids": sorted(self._applied_channels),
                    "safety_action_history_size": len(self.safety_controller.history)
                    if self.safety_controller is not None
                    else 0,
                    "target_corridor": snapshot_profile.target_corridor.to_dict(),
                },
                notes="Post-learning snapshot lock after initial correction pass",
            )

    def _apply_hpf(
        self,
        ch: int,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Set HPF cutoff using spectral band energy metrics.

        Metrics used:
        - spectral.band_energy['sub'] (20-60 Hz)
        - spectral.band_energy['bass'] (60-250 Hz)
        - spectral.centroid_hz — low centroid = real low content
        """
        info = self.channels[ch]
        base_freq = INSTRUMENT_HPF.get(preset, 80.0)
        adaptive_freq = base_freq

        m = info.metrics
        if m:
            sub_e = m.spectral.band_energy.get('sub', -100.0)
            bass_e = m.spectral.band_energy.get('bass', -100.0)
            low_mid_e = m.spectral.band_energy.get('low_mid', -100.0)
            centroid = m.spectral.centroid_hz

            # Convert from dB to linear for ratio comparison
            sub_lin = 10 ** (sub_e / 20.0) if sub_e > -90 else 0
            bass_lin = 10 ** (bass_e / 20.0) if bass_e > -90 else 0
            lm_lin = 10 ** (low_mid_e / 20.0) if low_mid_e > -90 else 0
            useful_ref = max(bass_lin, lm_lin, 1e-10)

            sub_ratio = sub_lin / useful_ref

            if sub_ratio < 0.05 and centroid > 500:
                # No useful sub, high centroid → raise HPF
                adaptive_freq = min(base_freq * 2.0, 400.0)
            elif sub_ratio < 0.15 and centroid > 200:
                adaptive_freq = min(base_freq * 1.4, 350.0)
            elif sub_ratio > 0.7 and preset in ("kick", "bass") and centroid < 200:
                # Real sub content on bass instrument → lower HPF
                adaptive_freq = max(base_freq * 0.6, 20.0)
            elif sub_ratio > 0.5 and centroid < 150:
                adaptive_freq = max(base_freq * 0.8, 25.0)

        raw_settings = self._snapshot_raw(info)
        current_hpf_enabled = self._as_bool(raw_settings.get("hpf_enabled"), False)
        current_hpf_freq = self._as_float(raw_settings.get("hpf_freq"), adaptive_freq)
        requested_freq = adaptive_freq
        if self.preserve_existing_processing and current_hpf_enabled:
            max_hpf_step = max(20.0, current_hpf_freq * 0.25)
            requested_freq = self._bounded_toward(
                current_hpf_freq,
                adaptive_freq,
                max_hpf_step,
            )

        try:
            if hasattr(self.mixer_client, 'set_hpf'):
                decision = self._execute_action(
                    HighPassAdjust(
                        channel_id=ch,
                        freq_hz=requested_freq,
                        enabled=True,
                        reason=f"HPF correction for {preset}",
                    ),
                    phase_guard_context=phase_guard_context,
                )
                if decision and decision.sent and current_hpf_enabled:
                    logger.info(
                        f"Ch {ch}: HPF {current_hpf_freq:.0f}→{requested_freq:.0f}Hz "
                        f"(analysis target {adaptive_freq:.0f}Hz)"
                    )
                elif decision and decision.sent and abs(requested_freq - base_freq) > 5:
                    logger.info(f"Ch {ch}: HPF={requested_freq:.0f}Hz (preset {base_freq:.0f}Hz)")
                elif decision and decision.sent:
                    logger.debug(f"Ch {ch}: HPF={requested_freq:.0f}Hz")
        except Exception as e:
            logger.warning(f"Ch {ch}: HPF failed: {e}")

    def _build_spectral_ceiling_eq_proposal(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
    ):
        if (
            self.spectral_ceiling_eq_analyzer is None
            or not self.spectral_ceiling_eq_config.enabled
            or self.audio_capture is None
        ):
            return None

        fft_size = int(self.autofoh_analysis_config.get("fft_size", 4096))
        window_samples = max(self.block_size * 4, fft_size * 2)
        try:
            samples = np.asarray(
                self.audio_capture.get_buffer(ch, window_samples),
                dtype=np.float32,
            ).reshape(-1)
        except Exception as exc:
            logger.debug("Ch %s: spectral ceiling audio capture failed: %s", ch, exc)
            return None
        if samples.size < max(1024, self.block_size // 2):
            return None

        lead_channel_ids = [
            lead_ch for lead_ch in self._lead_channel_ids()
            if lead_ch != ch and self.channels.get(lead_ch) is not None
        ]
        lead_confidence = 0.0
        if lead_channel_ids:
            lead_confidence = max(
                float(self.channels[lead_ch].classification_confidence)
                for lead_ch in lead_channel_ids
            )

        role = info.source_role or preset
        return self.spectral_ceiling_eq_analyzer.analyze(
            samples,
            instrument_role=role,
            sample_rate=self.sample_rate,
            track_id=info.name or f"Ch {ch}",
            role_confidence=float(info.classification_confidence or 0.0),
            lead_vocal_active=bool(lead_channel_ids),
            lead_vocal_confidence=lead_confidence,
        )

    def _log_spectral_ceiling_eq_proposal(
        self,
        proposal,
        merge_report: Optional[Dict[str, Any]],
    ) -> None:
        if proposal is None:
            return
        if self.spectral_ceiling_eq_config.log_verbose:
            logger.info("%s", format_spectral_ceiling_log(proposal, merge_report))
        self._log_autofoh_event(
            "spectral_ceiling_eq",
            channel=proposal.track_id,
            proposal=proposal.to_dict(),
            merge_report=merge_report or {},
        )

    def _apply_eq(
        self,
        ch: int,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Apply adaptive EQ using spectral metrics.

        Metrics used:
        - spectral.mud_ratio — excess 200-500Hz energy
        - spectral.presence_ratio — 2-5kHz clarity
        - spectral.brightness — high-frequency content
        - spectral.warmth — 200-800Hz warmth
        - spectral.spectral_tilt_db — overall spectrum slope
        - spectral.band_energy (7 bands) — per-band energy
        - spectral.flatness — tonal vs noise character
        """
        info = self.channels[ch]
        eq_bands = INSTRUMENT_EQ_PRESETS.get(preset)
        if not eq_bands:
            return

        adapted_bands = list(eq_bands)
        m = info.metrics

        if m and m.spectral.centroid_hz > 0:
            sm = m.spectral

            for i, (freq, gain, q) in enumerate(adapted_bands):
                adapted_gain = gain

                # Band 1: Low end (typically < 200 Hz)
                if freq <= 150:
                    if preset in ("kick", "bass"):
                        # Boost sub only if it's actually weak
                        sub_e = sm.band_energy.get('sub', -80)
                        bass_e = sm.band_energy.get('bass', -80)
                        if sub_e < bass_e - 10:
                            adapted_gain = max(gain, gain + 1.5)
                    else:
                        # For non-bass instruments, cut more if muddy
                        if sm.warmth > 0.4:
                            adapted_gain = min(gain, gain - 2.0)

                # Band 2: Low-mid / mud region (200-600 Hz)
                elif 150 < freq <= 600:
                    if sm.mud_ratio > 0.25:
                        mud_excess = (sm.mud_ratio - 0.15) * 20
                        adapted_gain = min(gain, gain - mud_excess * 0.5)
                    elif sm.mud_ratio < 0.05 and sm.warmth < 0.1:
                        adapted_gain = max(gain, gain + 1.0)

                # Band 3: Presence / clarity (1.5-5 kHz)
                elif 1500 <= freq <= 5000:
                    if sm.presence_ratio > 0.3:
                        excess = (sm.presence_ratio - 0.2) * 15
                        adapted_gain = min(gain, gain - excess * 0.4)
                    elif sm.presence_ratio < 0.08:
                        deficit = (0.15 - sm.presence_ratio) * 20
                        adapted_gain = max(gain, gain + deficit * 0.5)
                    if sm.flatness > 0.7 and gain > 0:
                        adapted_gain = gain * 0.5

                # Band 4: Air / brightness (6+ kHz)
                elif freq >= 6000:
                    if sm.brightness > 0.25:
                        bright_excess = (sm.brightness - 0.15) * 15
                        adapted_gain = min(gain, gain - bright_excess * 0.4)
                    elif sm.brightness < 0.03:
                        adapted_gain = max(gain, gain + 1.5)

                # Spectral tilt compensation
                if sm.spectral_tilt_db < -5.0 and freq >= 2000:
                    adapted_gain += min(2.0, abs(sm.spectral_tilt_db + 5) * 0.3)
                elif sm.spectral_tilt_db > 3.0 and freq <= 500:
                    adapted_gain += min(1.5, (sm.spectral_tilt_db - 3) * 0.2)

                adapted_gain = max(-8.0, min(8.0, round(adapted_gain, 1)))
                adapted_bands[i] = (freq, adapted_gain, q)

        spectral_proposal = self._build_spectral_ceiling_eq_proposal(ch, info, preset)
        spectral_merge_report = None
        if spectral_proposal is not None:
            adapted_bands, spectral_merge_report = merge_spectral_proposal_into_eq_bands(
                adapted_bands,
                spectral_proposal,
            )
            self._log_spectral_ceiling_eq_proposal(
                spectral_proposal,
                spectral_merge_report,
            )

        try:
            sent_any = False
            raw_settings = self._snapshot_raw(info)
            existing_eq_on = self._as_bool(raw_settings.get("eq_on"), False)
            existing_bands = raw_settings.get("eq_bands") or []
            if not existing_eq_on and hasattr(self.mixer_client, "set_eq_on"):
                self.mixer_client.set_eq_on(ch, 1)
            for band_idx, (freq, gain, q) in enumerate(adapted_bands, start=1):
                if band_idx > 4:
                    break
                target_freq = freq
                target_q = q
                if self.preserve_existing_processing and existing_eq_on and len(existing_bands) >= band_idx:
                    existing_band = existing_bands[band_idx - 1]
                    if len(existing_band) >= 3:
                        existing_freq = self._as_float(existing_band[0], freq)
                        existing_gain = self._as_float(existing_band[1], 0.0)
                        existing_q = self._as_float(existing_band[2], q)
                        ratio = max(existing_freq, freq) / max(1.0, min(existing_freq, freq))
                        if abs(existing_gain) > 0.1 and ratio <= 1.8:
                            target_freq = existing_freq
                            target_q = existing_q
                        elif abs(existing_gain) > 0.1:
                            logger.info(
                                f"Ch {ch} '{info.name}': EQ band {band_idx} "
                                f"retuned from {existing_freq:.0f}Hz to {freq:.0f}Hz "
                                "because the existing band does not match the requested problem range"
                            )
                decision = self._execute_action(
                    ChannelEQMove(
                        channel_id=ch,
                        band=band_idx,
                        freq_hz=target_freq,
                        gain_db=gain,
                        q=target_q,
                        reason=f"EQ correction for {preset}",
                    ),
                    phase_guard_context=phase_guard_context,
                )
                if decision and decision.sent:
                    sent_any = True

            info.eq_applied = sent_any
            changes = []
            for i, ((_, og, _), (_, ag, _)) in enumerate(zip(eq_bands, adapted_bands)):
                if abs(og - ag) > 0.3:
                    changes.append(f"B{i+1}: {og:+.1f}→{ag:+.1f}dB")
            if sent_any and changes:
                logger.info(f"Ch {ch}: EQ adapted: {', '.join(changes)}")
            elif sent_any:
                logger.debug(f"Ch {ch}: EQ preset '{preset}' (no adaptation)")
        except Exception as e:
            logger.warning(f"Ch {ch}: EQ failed: {e}")

    def _apply_gain_correction(self, ch: int, info: ChannelInfo, preset: str):
        """Compute LUFS-based gain correction and fold it into fader offset.

        NOTE: This does NOT call set_gain() — that's used by input gain
        staging (preamp trim). Calling set_gain() twice would overwrite
        the preamp setting. Instead, the LUFS correction is stored in
        info.gain_correction_db and applied via the fader position.

        Metrics used:
        - level.lufs_integrated — gated loudness (most accurate)
        - level.lufs_short_term — fallback
        - level.true_peak_dbtp — headroom check
        - level.loudness_range_lu — wide LRA → gentler correction
        """
        target_lufs = INSTRUMENT_TARGET_LUFS.get(preset, -23.0)

        m = info.metrics
        if m and m.level.lufs_integrated > -90:
            current_lufs = m.level.lufs_integrated
            true_peak = m.level.true_peak_dbtp
            lra = m.level.loudness_range_lu
        elif m and m.level.lufs_short_term > -90:
            current_lufs = m.level.lufs_short_term
            true_peak = m.level.true_peak_dbtp
            lra = 0.0
        else:
            current_lufs = self.audio_capture.get_lufs(ch, 48000)
            true_peak = info.peak_db
            lra = 0.0

        info.lufs = current_lufs

        if current_lufs <= -90.0:
            info.gain_correction_db = 0.0
            return

        diff = target_lufs - current_lufs

        scale = 1.0
        if lra > 15.0:
            scale = 0.7
        elif lra > 10.0:
            scale = 0.85

        correction = diff * scale
        correction = max(-GAIN_CORRECTION_MAX_DB, min(GAIN_CORRECTION_MAX_DB, correction))

        if true_peak + correction > -1.0:
            correction = -1.0 - true_peak
            correction = max(-GAIN_CORRECTION_MAX_DB, correction)

        info.gain_correction_db = correction

        if abs(correction) > 0.5:
            logger.info(
                f"Ch {ch} '{info.name}': LUFS correction {correction:+.1f}dB "
                f"(LUFS_I={current_lufs:.1f}→{target_lufs:.0f}, "
                f"TP={true_peak:.1f}dBTP, LRA={lra:.1f}LU) "
                f"→ applied via fader"
            )

    def _apply_fader(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Compute fader from LUFS balance, gain correction, and crest factor.

        The fader incorporates two components:
        1. Base position from instrument preset
        2. LUFS gain correction (computed by _apply_gain_correction)
        3. Inter-channel balance adjustment based on measured LUFS
        4. Crest factor compensation (peaky signals sound quieter)

        Metrics used:
        - level.lufs_integrated of all channels — relative balance
        - level.crest_factor_db — perceptual loudness compensation
        - gain_correction_db — LUFS-based correction folded into fader
        """
        fader_levels = {
            "kick": -5.0, "snare": -5.0, "tom": -8.0, "hihat": -12.0,
            "ride": -12.0, "cymbals": -12.0, "overheads": -10.0,
            "room": -15.0, "bass": -5.0, "electricGuitar": -8.0,
            "acousticGuitar": -8.0, "accordion": -8.0, "synth": -8.0,
            "playback": -10.0, "leadVocal": -3.0, "backVocal": -8.0,
        }
        base_fader = fader_levels.get(preset, -10.0)
        current_fader = self._as_float(
            self._snapshot_value(info, "fader_db", None),
            info.fader_db if info.fader_db > -140 else base_fader,
        )
        is_muted = self._as_bool(self._snapshot_value(info, "muted", False), False)

        if self.preserve_existing_processing and (is_muted or current_fader <= -90.0):
            info.fader_db = current_fader
            logger.info(
                f"Ch {ch} '{info.name}': fader correction skipped "
                f"(current={current_fader:.1f}dB, muted={is_muted})"
            )
            return

        # Start with the operator's current fader and apply only a correction.
        fader_db = current_fader + info.gain_correction_db

        # Collect LUFS from all active channels for relative balance
        all_lufs: Dict[int, Tuple[float, str]] = {}
        for c, ci in self.channels.items():
            if ci.has_signal:
                ci_lufs = -100.0
                if ci.metrics and ci.metrics.level.lufs_integrated > -90:
                    ci_lufs = ci.metrics.level.lufs_integrated
                elif ci.lufs > -90:
                    ci_lufs = ci.lufs
                if ci_lufs > -90:
                    all_lufs[c] = (ci_lufs, ci.preset or "custom")

        my_entry = all_lufs.get(ch)
        if my_entry is not None and len(all_lufs) > 1:
            my_lufs = my_entry[0]

            # Find reference: the loudest channel's LUFS and its target
            ref_ch = max(all_lufs, key=lambda c: all_lufs[c][0])
            ref_lufs, ref_preset = all_lufs[ref_ch]
            ref_target = INSTRUMENT_TARGET_LUFS.get(ref_preset, -23.0)

            # How far is this channel from where it should be relative to ref?
            my_target = INSTRUMENT_TARGET_LUFS.get(preset, -23.0)
            expected_diff = my_target - ref_target
            actual_diff = my_lufs - ref_lufs
            deviation = actual_diff - expected_diff

            # Crest factor compensation
            m = info.metrics
            crest_adj = 0.0
            if m and m.level.crest_factor_db > 18.0:
                crest_adj = 1.0
            elif m and m.level.crest_factor_db < 6.0:
                crest_adj = -1.0

            balance_adj = max(-4.0, min(4.0, -deviation * 0.5 + crest_adj))
            fader_db += balance_adj

        fader_db = max(-144.0, min(0.0, fader_db))
        previous_fader_db = current_fader

        try:
            decision = self._execute_action(
                ChannelFaderMove(
                    channel_id=ch,
                    target_db=fader_db,
                    is_lead=info.source_role == "lead_vocal",
                    reason=f"Fader target for {preset}",
                ),
                phase_guard_context=phase_guard_context,
            )
            if decision and decision.sent:
                info.fader_db = getattr(decision.action, "target_db", fader_db)
            else:
                info.fader_db = previous_fader_db

            total_adj = fader_db - current_fader
            if decision and decision.sent and abs(total_adj) > 0.5:
                logger.info(
                    f"Ch {ch} '{info.name}': fader {current_fader:.1f}→{fader_db:.1f}dB "
                    f"(preset_ref={base_fader:.1f}, lufs_corr={info.gain_correction_db:+.1f}, "
                    f"lufs={info.lufs:.1f})"
                )
            elif decision and decision.sent:
                logger.info(f"Ch {ch} '{info.name}': fader={fader_db:.1f}dB")
        except Exception as e:
            logger.warning(f"Ch {ch}: fader set failed: {e}")

    # ── 5d. Input gain staging ───────────────────────────────────

    def _apply_input_gain(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Set preamp gain using true peak and crest factor.

        Metrics used:
        - level.true_peak_dbtp — inter-sample accurate peak level
        - level.rms_db — average signal level
        - level.crest_factor_db — peak/RMS ratio (high = transient, low = sustained)
        - dynamics.dynamic_range_db — signal variability
        """
        m = info.metrics
        if m and m.level.true_peak_dbtp > -90:
            true_peak = m.level.true_peak_dbtp
            rms = m.level.rms_db
            crest = m.level.crest_factor_db
        else:
            true_peak = info.peak_db
            rms = info.rms_db
            crest = true_peak - rms if rms > -90 else 0

        if true_peak <= -90.0:
            return

        target_peak = INSTRUMENT_TARGET_INPUT_DB.get(preset, -12.0)

        # For high-crest signals (drums), target peak can be higher
        # because perceived loudness is lower
        if crest > 18.0:
            target_peak += 3.0
        elif crest < 6.0:
            target_peak -= 2.0

        diff = target_peak - true_peak
        correction = max(-20.0, min(20.0, diff))

        if abs(correction) < 1.0:
            return

        # Safety: true peak after correction must not exceed -1 dBTP
        if true_peak + correction > -1.0:
            correction = -1.0 - true_peak

        # Absolute gain safety clamp (per .cursorrules: gain ≤ +60 dB)
        SAFE_MAX_GAIN_DB = 12.0
        correction = max(-SAFE_MAX_GAIN_DB, min(SAFE_MAX_GAIN_DB, correction))
        current_gain = self._as_float(
            self._snapshot_value(info, "gain_db", None),
            self._as_float(getattr(info.original_snapshot, "gain_db", 0.0), 0.0)
            if info.original_snapshot else 0.0,
        )
        target_gain = max(-SAFE_MAX_GAIN_DB, min(SAFE_MAX_GAIN_DB, current_gain + correction))
        effective_correction = target_gain - current_gain

        if abs(effective_correction) < 1.0:
            return
        if effective_correction > 0.0 and not self.allow_input_trim_boost:
            logger.info(
                f"Ch {ch} '{info.name}': input trim boost skipped "
                f"({current_gain:+.1f}→{target_gain:+.1f}dB requested; "
                "live policy avoids reacting to bleed)"
            )
            return
        if (
            effective_correction > 0.0
            and info.original_snapshot is not None
            and (
                info.original_snapshot.muted
                or info.original_snapshot.fader_db <= -50.0
                or info.classification_confidence < self.minimum_auto_apply_classification_confidence
            )
        ):
            logger.info(
                f"Ch {ch} '{info.name}': input trim boost skipped "
                "(muted/parked or low-confidence source during line check)"
            )
            return

        try:
            decision = self._execute_action(
                ChannelGainMove(
                    channel_id=ch,
                    target_db=target_gain,
                    reason=f"Input gain correction for {preset}",
                ),
                phase_guard_context=phase_guard_context,
            )
            if decision and decision.sent:
                logger.info(
                    f"Ch {ch} '{info.name}': input gain {current_gain:+.1f}→{target_gain:+.1f}dB "
                    f"({effective_correction:+.1f}dB correction; "
                    f"TP={true_peak:.1f}dBTP, RMS={rms:.1f}, "
                    f"crest={crest:.1f}, target={target_peak:.0f}dB)"
                )
        except Exception as e:
            logger.warning(f"Ch {ch}: input gain failed: {e}")

    # ── 5e. Phase / polarity detection ────────────────────────

    def _detect_and_fix_phase(self):
        """Detect phase issues using inter-channel metrics.

        Metrics used (from compare_channels):
        - cross_correlation — GCC-PHAT peak value and sign
        - delay_ms — time offset between channels
        - coherence — frequency-domain coherence (high = correlated)
        - spectral_similarity — cosine similarity of spectra
        - phase_inverted — detected polarity inversion
        - level_difference_db — for weighting decisions
        """
        pairs = self._find_correlated_pairs()
        if not pairs:
            logger.info("Phase check: no correlated pairs found")
            return

        for ch_a, ch_b, pair_type in pairs:
            try:
                buf_a = self.audio_capture.get_buffer(ch_a, self.sample_rate * 2)
                buf_b = self.audio_capture.get_buffer(ch_b, self.sample_rate * 2)

                if len(buf_a) < 4096 or len(buf_b) < 4096:
                    continue

                # Use the full inter-channel comparison
                icm = compare_channels(
                    buf_a, buf_b, self.sample_rate, ch_a, ch_b
                )

                info_a = self.channels.get(ch_a)
                info_b = self.channels.get(ch_b)
                name_a = info_a.name if info_a else f"Ch {ch_a}"
                name_b = info_b.name if info_b else f"Ch {ch_b}"

                logger.info(
                    f"Phase: {name_a} ↔ {name_b}: "
                    f"corr={icm.cross_correlation:.3f} "
                    f"delay={icm.delay_ms:.2f}ms "
                    f"coherence={icm.coherence:.3f} "
                    f"spec_sim={icm.spectral_similarity:.3f} "
                    f"level_diff={icm.level_difference_db:.1f}dB"
                )

                # Only act if signals are actually related (high coherence)
                if icm.coherence < 0.2 and icm.spectral_similarity < 0.5:
                    logger.debug(f"Phase: skipping — low coherence/similarity")
                    continue

                # Phase inversion detected
                if icm.phase_inverted:
                    raw_b = self._snapshot_raw(info_b) if info_b else {}
                    current_polarity = self._as_bool(
                        raw_b.get("polarity_inverted"),
                        False,
                    )
                    if self.preserve_existing_processing and current_polarity:
                        logger.info(
                            f"Phase: Ch {ch_b} polarity already inverted; preserving current state"
                        )
                        continue
                    logger.warning(
                        f"Phase: {name_b} is inverted (corr={icm.cross_correlation:.3f}), "
                        f"correcting polarity through safety layer"
                    )
                    self._execute_action(
                        PolarityAdjust(
                            channel_id=ch_b,
                            inverted=True,
                            reason=(
                                f"Phase polarity correction for {pair_type}: "
                                f"corr={icm.cross_correlation:.3f}"
                            ),
                        ),
                        evaluation_context={
                            "problem_type": "drum_phase_alignment",
                            "expected_effect": "Improve correlated drum polarity without resetting processing",
                            "pair": [ch_a, ch_b],
                            "pair_type": pair_type,
                        },
                    )

                # Time alignment
                elif abs(icm.delay_ms) > 0.1 and icm.coherence > 0.3:
                    align_ch = ch_b if icm.delay_samples > 0 else ch_a
                    align_info = self.channels.get(align_ch)
                    align_raw = self._snapshot_raw(align_info) if align_info else {}
                    current_delay_enabled = self._as_bool(
                        align_raw.get("delay_enabled"),
                        False,
                    )
                    current_delay_ms = self._as_float(
                        align_raw.get("delay_ms"),
                        0.0,
                    )
                    measured_delay_ms = abs(float(icm.delay_ms))
                    delay_target_ms = current_delay_ms + measured_delay_ms
                    logger.info(
                        f"Phase: correcting Ch {align_ch} delay "
                        f"{current_delay_ms:.2f}→{delay_target_ms:.2f}ms "
                        f"(measured={measured_delay_ms:.2f}ms, "
                        f"enabled={current_delay_enabled}, coherence={icm.coherence:.3f})"
                    )
                    self._execute_action(
                        DelayAdjust(
                            channel_id=align_ch,
                            delay_ms=delay_target_ms,
                            enabled=True,
                            reason=(
                                f"Drum phase delay alignment for {pair_type}: "
                                f"{measured_delay_ms:.2f}ms residual"
                            ),
                        ),
                        evaluation_context={
                            "problem_type": "drum_phase_alignment",
                            "expected_effect": "Reduce correlated drum timing offset without clearing existing delay",
                            "pair": [ch_a, ch_b],
                            "pair_type": pair_type,
                            "measured_delay_ms": measured_delay_ms,
                        },
                    )

            except Exception as e:
                logger.warning(f"Phase check error for pair ({ch_a}, {ch_b}): {e}")

    def _find_correlated_pairs(self) -> List[Tuple[int, int, str]]:
        """Find channel pairs that should be phase-correlated.

        Uses source-aware drum rules from the offline mixing workflow:
        kick/snare/overhead/room mics are aligned within their own source
        family, while different toms are not treated as one correlated source.
        """
        pairs = []
        preset_channels: Dict[str, List[int]] = {}

        for ch, info in self.channels.items():
            if (
                info.has_signal
                and info.preset
                and ch not in self.master_reference_channels
            ):
                preset_channels.setdefault(info.preset, []).append(ch)

        drum_pair_presets = {"kick", "snare", "overheads", "room"}
        for preset in drum_pair_presets:
            channels = sorted(preset_channels.get(preset, []))
            if len(channels) < 2:
                continue
            reference = channels[0]
            for other in channels[1:]:
                pairs.append((reference, other, preset))

        phase_pair_presets = set(STEREO_PAIR_PRESETS)
        for preset, channels in preset_channels.items():
            if len(channels) < 2:
                continue
            if preset in drum_pair_presets or preset == "tom":
                continue
            if preset not in phase_pair_presets:
                continue
            channels.sort()
            # Pair adjacent channels
            for i in range(0, len(channels) - 1, 2):
                pairs.append((channels[i], channels[i + 1], preset))

        return pairs

    # ── 5f. Pan positioning ──────────────────────────────────────

    def _apply_pan(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Set pan position based on instrument type."""
        if not hasattr(self.mixer_client, 'set_pan'):
            return

        pan = INSTRUMENT_PAN.get(preset, 0.0)
        raw_settings = self._snapshot_raw(info)
        if self.preserve_existing_processing and "pan" in raw_settings:
            current_pan = self._as_float(raw_settings.get("pan"), 0.0)
            logger.info(
                f"Ch {ch} '{info.name}': pan preserved at {current_pan:+.0f}"
            )
            return

        # Smart panning for stereo pairs and multiple instruments
        preset_channels = [
            c for c, i in self.channels.items()
            if i.preset == preset and i.has_signal
        ]
        preset_channels.sort()

        if len(preset_channels) >= 2:
            idx = preset_channels.index(ch)
            if preset in STEREO_PAIR_PRESETS:
                # Hard stereo pair: L/R
                pan = -60.0 if idx % 2 == 0 else 60.0
            elif preset == "tom":
                # Spread toms L to R
                spread = 50.0
                n = len(preset_channels)
                if n > 1:
                    pan = -spread + (2.0 * spread * idx / (n - 1))
                else:
                    pan = 0.0
            elif preset == "backVocal":
                spread = 35.0
                n = len(preset_channels)
                if n == 2:
                    pan = -spread if idx == 0 else spread
                else:
                    pan = -spread + (2.0 * spread * idx / max(1, n - 1))

        try:
            decision = self._execute_action(
                PanAdjust(
                    channel_id=ch,
                    pan=pan,
                    reason=f"Pan position for {preset}",
                ),
                phase_guard_context=phase_guard_context,
            )
            if decision and decision.sent:
                logger.info(f"Ch {ch} '{info.name}': pan = {pan:+.0f}")
        except Exception as e:
            logger.warning(f"Ch {ch}: pan set failed: {e}")

    # ── 5g. Gate / Expander ──────────────────────────────────────

    def _apply_gate(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Enable a bounded gate on close drum mics before tonal decisions."""
        if not info.auto_corrections_enabled:
            return
        if self._stem_ml_corrections_frozen(info):
            return
        if not self.auto_gate_close_mics:
            return
        if preset not in INSTRUMENT_GATE_PRESETS:
            return
        if not hasattr(self.mixer_client, "set_gate"):
            return

        base_threshold, base_range, base_attack, base_hold, base_release = INSTRUMENT_GATE_PRESETS[preset]
        threshold = base_threshold
        range_db = base_range
        attack = base_attack
        hold = base_hold
        release = base_release

        m = info.metrics
        if m and m.level.rms_db > -90 and m.level.true_peak_dbtp > -90:
            # Keep the threshold below real hits, but above likely bleed.
            threshold = min(m.level.true_peak_dbtp - 10.0, m.level.rms_db + 6.0)
            threshold = max(base_threshold - 10.0, min(base_threshold + 8.0, threshold))
            if m.level.crest_factor_db > 18.0:
                attack = min(attack, 1.5)
                hold = max(hold, 90.0)

        raw_settings = self._snapshot_raw(info)
        if self.preserve_existing_processing and self._as_bool(raw_settings.get("gate_enabled"), False):
            current_threshold = self._as_float(raw_settings.get("gate_threshold_db"), threshold)
            current_range = self._as_float(raw_settings.get("gate_range_db"), range_db)
            threshold = self._bounded_toward(current_threshold, threshold, 4.0)
            range_db = self._bounded_toward(current_range, range_db, 6.0)

        try:
            decision = self._execute_action(
                GateAdjust(
                    channel_id=ch,
                    threshold_db=round(threshold, 1),
                    range_db=round(range_db, 1),
                    attack_ms=round(attack, 1),
                    hold_ms=round(hold, 1),
                    release_ms=round(release, 1),
                    enabled=True,
                    reason=f"Gate profile for {preset}",
                ),
                phase_guard_context=phase_guard_context,
            )
            if decision and decision.sent:
                logger.info(
                    f"Ch {ch} '{info.name}': gate on thr={threshold:.0f}dB "
                    f"range={range_db:.0f}dB atk={attack:.0f}ms hold={hold:.0f}ms rel={release:.0f}ms"
                )
        except Exception as e:
            logger.warning(f"Ch {ch}: gate failed: {e}")

    # ── 5h. Compressor ───────────────────────────────────────────

    @staticmethod
    def _normalise_gain_reduction_db(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            gr = abs(float(value))
        except (TypeError, ValueError):
            return None
        if not np.isfinite(gr):
            return None
        return max(0.0, min(24.0, gr))

    def _compressor_peak_db_for_headroom(self, metrics: Optional[ChannelMetrics]) -> Optional[float]:
        if metrics is None:
            return None
        level = getattr(metrics, "level", None)
        if level is None:
            return None
        for attr in ("true_peak_dbtp", "peak_db"):
            value = getattr(level, attr, None)
            try:
                peak_db = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(peak_db) and peak_db > -90.0:
                return peak_db
        return None

    def _estimate_compressor_gr_from_signal(
        self,
        raw_settings: Dict[str, Any],
        metrics: Optional[ChannelMetrics],
        model: str,
    ) -> Optional[float]:
        if metrics is None:
            return None
        level = getattr(metrics, "level", None)
        if level is None:
            return None
        peak_db = self._compressor_peak_db_for_headroom(metrics)
        rms_db = self._as_float(getattr(level, "rms_db", None), -100.0)
        if peak_db is None or (peak_db < -55.0 and rms_db < -70.0):
            return None

        model = str(model or "").upper()
        threshold = self._as_float(raw_settings.get("compressor_threshold_db"), None)
        ratio = self._as_float(raw_settings.get("compressor_ratio"), 2.0)
        mix = self._as_float(raw_settings.get("compressor_mix_pct"), 100.0)
        mix_scale = max(0.0, min(1.0, mix / 100.0))
        detector = str(raw_settings.get("compressor_detector") or "RMS").upper()

        if model in {"COMP", "STD", "B560", "RED3"} and threshold is not None and ratio > 1.01:
            detector_level = peak_db if detector == "PEAK" else rms_db
            over_threshold = detector_level - threshold
            if over_threshold <= 0.0:
                return 0.0
            return max(0.0, over_threshold * (1.0 - (1.0 / ratio)) * mix_scale)

        # BDX160 exposes threshold as a unitless 0.01..5 parameter. Without a
        # read-only GR meter, use a conservative audition value only when the
        # signal is clearly active and the control is near maximum sensitivity.
        if model == "B160" and threshold is not None and threshold <= 0.05 and peak_db > -12.0:
            return 1.5 * mix_scale

        return None

    def _calculate_compressor_makeup_target(
        self,
        *,
        current_makeup_db: Any,
        gain_reduction_db: Any,
        metrics: Optional[ChannelMetrics],
        assume_target_gr: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        current_makeup = self._as_float(current_makeup_db, 0.0)
        current_makeup = max(-6.0, min(12.0, current_makeup))

        measured_gr = self._normalise_gain_reduction_db(gain_reduction_db)
        gr_for_compensation = measured_gr or 0.0
        if gr_for_compensation <= 0.25 and assume_target_gr:
            gr_for_compensation = max(0.0, self.compressor_gr_target_db)

        overcompressed = (
            measured_gr is not None
            and measured_gr > self.compressor_gr_max_db
        )
        if overcompressed:
            gr_for_compensation = min(gr_for_compensation, self.compressor_gr_target_db)

        desired_makeup = gr_for_compensation * max(
            0.0,
            min(1.0, self.compressor_makeup_compensation_ratio),
        )
        desired_makeup = max(0.0, min(self.compressor_makeup_max_db, desired_makeup))

        peak_db = self._compressor_peak_db_for_headroom(metrics)
        headroom_limited = False
        if peak_db is not None and desired_makeup > current_makeup:
            available_headroom = self.compressor_makeup_true_peak_ceiling_db - peak_db
            if available_headroom <= 0.0:
                desired_makeup = current_makeup
                headroom_limited = True
            else:
                allowed_makeup = current_makeup + available_headroom
                if desired_makeup > allowed_makeup:
                    desired_makeup = allowed_makeup
                    headroom_limited = True

        # Preserve intentionally hotter makeup unless it exceeds our live cap.
        if desired_makeup < current_makeup and current_makeup <= self.compressor_makeup_max_db:
            desired_makeup = current_makeup

        target_makeup = self._bounded_toward(
            current_makeup,
            desired_makeup,
            max(0.1, self.compressor_makeup_max_step_db),
        )
        details = {
            "current_makeup_db": round(current_makeup, 2),
            "target_makeup_db": round(target_makeup, 2),
            "desired_makeup_db": round(desired_makeup, 2),
            "measured_gr_db": None if measured_gr is None else round(measured_gr, 2),
            "gr_used_for_compensation_db": round(gr_for_compensation, 2),
            "compensation_ratio": round(self.compressor_makeup_compensation_ratio, 2),
            "headroom_peak_db": None if peak_db is None else round(peak_db, 2),
            "true_peak_ceiling_db": round(self.compressor_makeup_true_peak_ceiling_db, 2),
            "headroom_limited": headroom_limited,
            "overcompressed": overcompressed,
            "source_rules": [
                "compression.reason_only",
                "compression.makeup_compensates_gain_reduction_safely",
                "dynamics.preserve_microdynamics_first",
                "automation.keep_effect_parameters_interpretable",
            ],
            "perceptual_shadow_enabled": self._perceptual_shadow_enabled(),
            "perceptual_backend": (
                getattr(self.perceptual_evaluator.backend, "name", "")
                if self.perceptual_evaluator is not None
                else ""
            ),
        }
        return target_makeup, details

    def _apply_compressor_makeup_adjustment(
        self,
        ch: int,
        info: ChannelInfo,
        target_makeup_db: float,
        details: Dict[str, Any],
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[SafetyDecision]:
        current_makeup = float(details.get("current_makeup_db", 0.0))
        if abs(target_makeup_db - current_makeup) < 0.1:
            logger.info(
                f"Ch {ch} '{info.name}': compressor makeup kept at {current_makeup:.1f}dB "
                f"(GR={details.get('measured_gr_db')}dB, headroom_limited={details.get('headroom_limited')})"
            )
            return None

        decision = self._execute_action(
            CompressorMakeupAdjust(
                channel_id=ch,
                makeup_db=round(target_makeup_db, 1),
                reason=(
                    "Compensate compressor gain reduction with bounded makeup gain; "
                    "source_rules=compression.makeup_compensates_gain_reduction_safely,"
                    "dynamics.preserve_microdynamics_first"
                ),
            ),
            evaluation_context={
                "expected_effect": "restore level lost to compression without bypassing headroom",
                **details,
            },
            phase_guard_context=phase_guard_context,
        )
        if decision and decision.sent:
            logger.info(
                f"Ch {ch} '{info.name}': compressor makeup "
                f"{current_makeup:.1f}→{target_makeup_db:.1f}dB, "
                f"GR={details.get('measured_gr_db')}dB, "
                f"rules={','.join(details.get('source_rules', []))}"
            )
            self._emit_observation(
                operation={
                    "action": "compressor_makeup",
                    "channel": ch,
                    "channel_name": info.name,
                    "message": (
                        f"Ch {ch} compressor makeup {current_makeup:.1f}→"
                        f"{target_makeup_db:.1f}dB"
                    ),
                    **details,
                }
            )
        return decision

    def _apply_compressor(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Apply compressor adapted to measured signal dynamics.

        Uses full metrics from SignalAnalyzer: crest factor, dynamic range,
        ADSR envelope, transient density/strength, and spectral flux to
        precisely adapt threshold, ratio, attack, and release.
        """
        if not hasattr(self.mixer_client, 'set_compressor'):
            return

        comp_params = INSTRUMENT_COMPRESSOR.get(preset)
        if not comp_params:
            return

        base_threshold, base_ratio, base_attack, base_release = comp_params
        threshold = base_threshold
        ratio = base_ratio
        attack = base_attack
        release = base_release
        raw_settings = self._snapshot_raw(info)
        current_comp_enabled = self._as_bool(
            raw_settings.get("compressor_enabled"),
            False,
        )
        current_model = str(raw_settings.get("compressor_model") or "").upper()
        existing_gr = None
        gr_getter = getattr(self.mixer_client, "get_compressor_gr", None)
        if gr_getter is not None:
            try:
                existing_gr = gr_getter(ch)
            except Exception:
                existing_gr = None
        existing_gr_db = self._normalise_gain_reduction_db(existing_gr)
        compressor_gr_source = "osc_meter" if existing_gr_db is not None else "unavailable"
        if existing_gr_db is None:
            estimated_gr = self._estimate_compressor_gr_from_signal(
                raw_settings,
                info.metrics,
                current_model,
            )
            if estimated_gr is not None:
                existing_gr = estimated_gr
                existing_gr_db = self._normalise_gain_reduction_db(estimated_gr)
                compressor_gr_source = "signal_estimate"

        m = info.metrics
        if m and m.level.rms_db > -80:
            rms_db = m.level.rms_db
            crest_factor = m.level.crest_factor_db
            dynamic_range = m.dynamics.dynamic_range_db
            measured_attack = m.dynamics.attack_time_ms
            measured_release = m.dynamics.release_time_ms
            transient_density = m.dynamics.transient_density
            spectral_flux = m.spectral.flux

            # Threshold: ~6dB above measured RMS, bounded by preset ±6
            threshold = rms_db + 6.0
            threshold = max(base_threshold - 6.0, min(base_threshold + 6.0, threshold))
            threshold = max(-50.0, min(-5.0, threshold))

            # Ratio: scale by crest factor
            if crest_factor > 20.0:
                ratio = min(base_ratio * 1.3, 10.0)
            elif crest_factor < 8.0:
                ratio = max(base_ratio * 0.7, 1.5)

            # Attack: use measured attack time as guide
            if measured_attack > 0 and measured_attack < 100:
                attack = max(1.0, measured_attack * 0.5)
            elif crest_factor > 15.0:
                attack = max(base_attack, 5.0)
            elif crest_factor < 6.0:
                attack = max(1.0, base_attack * 0.7)

            # If high transient density (drums), preserve transients
            if transient_density > 4.0:
                attack = max(attack, 3.0)

            # Release: use measured release or adapt by dynamic range
            if measured_release > 10 and measured_release < 2000:
                release = max(30.0, measured_release * 0.7)
            elif dynamic_range > 15.0:
                release = min(base_release * 1.3, 500.0)
            elif dynamic_range < 6.0:
                release = max(base_release * 0.7, 30.0)

            # High spectral flux → more dynamic signal → longer release
            if spectral_flux > 0.5:
                release = min(release * 1.2, 600.0)

            logger.debug(
                f"Ch {ch}: comp adaptation: crest={crest_factor:.1f}dB "
                f"DR={dynamic_range:.1f}dB atk_measured={measured_attack:.0f}ms "
                f"trans_density={transient_density:.1f}/s flux={spectral_flux:.3f}"
            )
        else:
            crest_factor = 0.0
            dynamic_range = 0.0

        if not current_comp_enabled and m and crest_factor < 8.0 and dynamic_range < 8.0:
            logger.info(
                f"Ch {ch} '{info.name}': compressor skipped; measured dynamics do not justify enabling it"
            )
            return

        current_makeup = self._as_float(raw_settings.get("compressor_makeup_db"), 0.0)
        makeup, makeup_details = self._calculate_compressor_makeup_target(
            current_makeup_db=current_makeup,
            gain_reduction_db=existing_gr,
            metrics=m,
            assume_target_gr=not current_comp_enabled and m is not None,
        )
        makeup_details["compressor_gr_source"] = compressor_gr_source

        if (
            self.preserve_existing_processing
            and current_comp_enabled
            and current_model
            and current_model not in {"COMP", "STD"}
        ):
            if existing_gr_db is not None and existing_gr_db > self.compressor_gr_max_db:
                current_threshold = self._as_float(
                    raw_settings.get("compressor_threshold_db"),
                    threshold,
                )
                target_threshold = current_threshold + min(
                    6.0,
                    existing_gr_db - self.compressor_gr_target_db,
                )
                if hasattr(self.mixer_client, "set_compressor_threshold"):
                    self.mixer_client.set_compressor_threshold(ch, round(target_threshold, 1))
                    logger.info(
                        f"Ch {ch} '{info.name}': {current_model} GR={existing_gr_db:.1f}dB; "
                        f"raising threshold {current_threshold:.1f}→{target_threshold:.1f}dB"
                    )
            else:
                logger.info(
                    f"Ch {ch} '{info.name}': preserving {current_model} compressor model; "
                    "generic compressor timing write skipped"
                )
            makeup_decision = self._apply_compressor_makeup_adjustment(
                ch,
                info,
                makeup,
                makeup_details,
                phase_guard_context=phase_guard_context,
            )
            if makeup_decision and makeup_decision.sent:
                info.compressor_applied = True
            return

        if self.preserve_existing_processing and current_comp_enabled:
            current_threshold = self._as_float(
                raw_settings.get("compressor_threshold_db"),
                threshold,
            )
            current_ratio = self._as_float(
                raw_settings.get("compressor_ratio"),
                ratio,
            )
            current_attack = self._as_float(
                raw_settings.get("compressor_attack_ms"),
                attack,
            )
            current_release = self._as_float(
                raw_settings.get("compressor_release_ms"),
                release,
            )
            if existing_gr_db is not None and existing_gr_db > self.compressor_gr_max_db:
                threshold = max(threshold, current_threshold + min(
                    6.0,
                    existing_gr_db - self.compressor_gr_target_db,
                ))
                ratio = min(ratio, max(1.5, current_ratio - 0.5))
            threshold = self._bounded_toward(current_threshold, threshold, 3.0)
            ratio = self._bounded_toward(current_ratio, ratio, 1.0)
            attack = self._bounded_toward(current_attack, attack, 10.0)
            release = self._bounded_toward(current_release, release, 75.0)

        try:
            decision = self._execute_action(
                CompressorAdjust(
                    channel_id=ch,
                    threshold_db=round(threshold, 1),
                    ratio=round(ratio, 1),
                    attack_ms=round(attack, 1),
                    release_ms=round(release, 1),
                    makeup_db=round(makeup, 1),
                    enabled=True,
                    reason=(
                        f"Compressor profile for {preset}; makeup compensates gain reduction "
                        "per source rule compression.makeup_compensates_gain_reduction_safely"
                    ),
                ),
                evaluation_context={
                    "expected_effect": (
                        "control dynamics and restore level lost to compression without "
                        "bypassing headroom"
                    ),
                    **makeup_details,
                },
                phase_guard_context=phase_guard_context,
            )
            info.compressor_applied = bool(decision and decision.sent)

            changes = []
            if abs(threshold - base_threshold) > 1.0:
                changes.append(f"thr: {base_threshold:.0f}→{threshold:.0f}")
            if abs(ratio - base_ratio) > 0.3:
                changes.append(f"ratio: {base_ratio:.1f}→{ratio:.1f}")
            if abs(attack - base_attack) > 2.0:
                changes.append(f"atk: {base_attack:.0f}→{attack:.0f}")
            if abs(release - base_release) > 10.0:
                changes.append(f"rel: {base_release:.0f}→{release:.0f}")
            if abs(makeup - current_makeup) > 0.2:
                changes.append(f"makeup: {current_makeup:.1f}→{makeup:.1f}")

            adapted = f" (adapted: {', '.join(changes)})" if changes else ""
            if decision and decision.sent:
                logger.info(
                    f"Ch {ch} '{info.name}': compressor thr={threshold:.0f}dB "
                    f"ratio={ratio:.1f}:1 atk={attack:.0f}ms rel={release:.0f}ms "
                    f"makeup={makeup:.1f}dB GR={makeup_details.get('measured_gr_db')}dB{adapted}"
                )
                self._emit_observation(
                    operation={
                        "action": "compressor",
                        "channel": ch,
                        "channel_name": info.name,
                        "message": (
                            f"Ch {ch} compressor: thr {threshold:.1f}dB, "
                            f"ratio {ratio:.1f}:1, makeup {makeup:.1f}dB"
                        ),
                        "threshold_db": round(threshold, 1),
                        "ratio": round(ratio, 1),
                        "attack_ms": round(attack, 1),
                        "release_ms": round(release, 1),
                        **makeup_details,
                    }
                )
        except Exception as e:
            logger.warning(f"Ch {ch}: compressor failed: {e}")

    # ── 5h. FX Sends (reverb / delay) ────────────────────────────

    def _apply_fx_sends(
        self,
        ch: int,
        info: ChannelInfo,
        preset: str,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ):
        """Set FX send levels using signal metrics.

        Metrics used:
        - level.lufs_integrated — loudness-based scaling
        - dynamics.dynamic_range_db — dynamic signals need less reverb
        - dynamics.sustain_level_db — sustained signals blend with reverb
        - spectral.brightness — bright signals → less reverb to avoid wash
        - spectral.warmth — warm signals → more reverb (natural fit)
        - dynamics.transient_density — percussive → shorter/less reverb
        """
        if not hasattr(self.mixer_client, 'set_send_level'):
            return
        if preset in {"kick", "snare", "tom"} and not self.close_mic_fx_sends_enabled:
            self._trim_unapproved_close_mic_fx_sends(
                ch,
                info,
                phase_guard_context=phase_guard_context,
            )
        if not self.auto_fx_sends_enabled:
            logger.info(
                f"Ch {ch} '{info.name}': automatic FX sends disabled for live safety"
            )
            return
        if info.channel in self.master_reference_channels or "MASTER" in info.stem_roles:
            logger.info(f"Ch {ch} '{info.name}': FX send skipped for master reference channel")
            return
        if preset in {"kick", "snare", "tom"} and not self.close_mic_fx_sends_enabled:
            logger.info(
                f"Ch {ch} '{info.name}': close-mic FX send skipped; use operator-approved drum bus/FX routing"
            )
            return

        sends = INSTRUMENT_FX_SENDS.get(preset, [])
        if not sends:
            return

        m = info.metrics
        adapted_sends = []

        for send_bus, base_level in sends:
            level = base_level

            if m:
                # Loudness scaling
                lufs_i = m.level.lufs_integrated
                if lufs_i > -16.0:
                    level -= 4.0
                elif lufs_i > -20.0:
                    level -= 2.0
                elif lufs_i < -35.0:
                    level += 3.0

                # Dynamic range: wide DR → less send (reverb masks dynamics)
                if m.dynamics.dynamic_range_db > 18.0:
                    level -= 2.0

                # Transient density: percussive → less reverb
                if m.dynamics.transient_density > 5.0:
                    level -= 2.0
                elif m.dynamics.transient_density < 1.0 and m.dynamics.sustain_level_db > -30:
                    level += 1.5

                # Brightness: bright → less reverb to keep clarity
                if m.spectral.brightness > 0.25:
                    level -= 2.0
                elif m.spectral.brightness < 0.05:
                    level += 1.0

                # Warmth: warm signals blend well with reverb
                if m.spectral.warmth > 0.3:
                    level += 1.0

            level = max(-40.0, min(-5.0, round(level, 1)))
            adapted_sends.append((send_bus, level))

        sent_buses = []
        for send_bus, level_db in adapted_sends:
            try:
                decision = self._execute_action(
                    SendLevelAdjust(
                        channel_id=ch,
                        send_bus=send_bus,
                        level_db=level_db,
                        reason=f"FX send for {preset}",
                    ),
                    phase_guard_context=phase_guard_context,
                )
                if decision and decision.sent:
                    sent_buses.append((send_bus, level_db))
            except Exception as e:
                logger.warning(f"Ch {ch}: send bus {send_bus} failed: {e}")

        if sent_buses:
            bus_desc = ", ".join(f"bus{b}={l:.0f}dB" for b, l in sent_buses)
            logger.info(f"Ch {ch} '{info.name}': FX sends: {bus_desc}")

    def _trim_unapproved_close_mic_fx_sends(
        self,
        ch: int,
        info: ChannelInfo,
        *,
        phase_guard_context: Optional[Dict[str, Any]] = None,
    ) -> List[SafetyDecision]:
        """Pull close drum mics out of non-monitor spatial FX sends."""
        raw_settings = self._snapshot_raw(info)
        active_sends = raw_settings.get("active_bus_sends") or []
        if not active_sends:
            return []

        decisions: List[SafetyDecision] = []
        for send in active_sends:
            try:
                bus_id = int(send.get("bus"))
            except (TypeError, ValueError):
                continue
            if self._is_monitor_bus(bus_id):
                continue
            bus_name = self.bus_snapshots.get(bus_id).name if bus_id in self.bus_snapshots else ""
            if not self._is_effect_bus(bus_id, bus_name):
                continue
            level_db = self._as_float(send.get("level_db"), -40.0)
            if level_db <= -39.5:
                continue
            decision = self._execute_action(
                SendLevelAdjust(
                    channel_id=ch,
                    send_bus=bus_id,
                    level_db=-40.0,
                    reason="Close drum mic routed to non-monitor spatial FX bus; trim send safely",
                ),
                evaluation_context={
                    "problem_type": "routing_cleanup",
                    "expected_effect": "Remove close-mic drum bleed from delay/reverb send without touching monitor buses",
                    "send_bus": bus_id,
                },
                phase_guard_context=phase_guard_context,
            )
            if decision is not None:
                decisions.append(decision)
        return decisions

    # ── 6. Continuous monitoring ─────────────────────────────────

    def _monitor_loop(self):
        """Continuous monitoring: feedback detection + cautious mix cleanup."""
        self.feedback_detector = FeedbackDetector(
            config=FeedbackDetectorConfig(
                sample_rate=self.sample_rate,
                fft_size=2048,
                persistence_frames=self.feedback_detector_persistence_frames,
                min_confidence=self.feedback_detector_min_confidence,
                peak_height_db=self.feedback_detector_peak_height_db,
                peak_prominence_db=self.feedback_detector_peak_prominence_db,
            ),
        )

        while not self._stop_event.is_set():
            try:
                for ch, info in self.channels.items():
                    if not info.has_signal:
                        peak = self.audio_capture.get_peak(ch, self.block_size * 4)
                        if peak > SIGNAL_THRESHOLD_DB:
                            info.has_signal = True
                            info.peak_db = peak
                            info.rms_db = self.audio_capture.get_rms(ch, self.block_size * 4)
                            info.lufs = self.audio_capture.get_lufs(ch, 0.4)
                            if not info.recognized:
                                self._classify_by_spectrum(ch, info)
                            logger.info(f"Monitor: new signal on Ch {ch} '{info.name}'")
                            self._apply_corrections_single(ch)
                        continue

                    samples = self.audio_capture.get_buffer(ch, 2048)
                    if len(samples) >= 2048:
                        events = self.feedback_detector.process(ch, samples)
                        for evt in events:
                            self._handle_feedback_event(ch, evt)

                self._evaluate_pending_actions()
                self._run_autofoh_monitor_analysis()

            except Exception as e:
                logger.debug(f"Monitor loop error: {e}")

            time.sleep(0.1)

    def _apply_corrections_single(self, ch: int):
        """Apply bounded processing corrections for a newly detected channel.

        Reads the current channel state first, then runs a short analysis pass
        and applies corrections relative to the preserved console snapshot.
        """
        info = self.channels.get(ch)
        if info is None or ch in self._applied_channels:
            return
        if not info.auto_corrections_enabled:
            self._log_processing_skip(ch, info, "single-channel auto-processing")
            return
        if self._stem_ml_corrections_frozen(info):
            self._log_processing_skip(ch, info, "MuQ stem freeze")
            return
        preset = info.preset or "custom"
        try:
            try:
                raw_settings = self.mixer_client.get_channel_settings(ch)
            except Exception as read_exc:
                logger.debug(f"Ch {ch}: could not read settings before single-channel correction: {read_exc}")
                raw_settings = {}
            fader_db = self._as_float(raw_settings.get("fader_db"), info.fader_db)
            snapshot = ChannelSnapshot(
                channel=ch,
                fader_db=fader_db,
                muted=self._as_bool(raw_settings.get("muted"), False),
                eq_bands=raw_settings.get("eq_bands"),
                hpf_freq=self._as_float(raw_settings.get("hpf_freq"), 20.0),
                hpf_enabled=self._as_bool(raw_settings.get("hpf_enabled"), False),
                gain_db=self._as_float(raw_settings.get("gain_db"), 0.0),
                had_processing=self._channel_has_existing_processing(raw_settings),
                raw_settings=raw_settings,
            )
            info.original_snapshot = snapshot
            info.fader_db = fader_db
            info.was_reset = False
            self._read_group_state_for_channels([ch])

            # Quick analysis pass (1.5s) for fresh metrics
            analyzer = SignalAnalyzer(ch, self.sample_rate, self.block_size)
            analysis_blocks = int(1.5 * self.sample_rate / self.block_size)
            interval = self.block_size / self.sample_rate
            for _ in range(analysis_blocks):
                samples = self.audio_capture.get_buffer(ch, self.block_size)
                if len(samples) >= self.block_size:
                    analyzer.process(samples)
                time.sleep(interval)
            info.metrics = analyzer.get_metrics()
            phase_guard_context = self._build_phase_target_guard_context(
                runtime_state=self.runtime_state
            )

            if self._control_allowed(info, "gain"):
                self._apply_input_gain(
                    ch,
                    info,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            if self._control_allowed(info, "hpf"):
                self._apply_hpf(
                    ch,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            self._apply_gate(
                ch,
                info,
                preset,
                phase_guard_context=phase_guard_context,
            )
            if self._control_allowed(info, "eq"):
                self._apply_eq(
                    ch,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            if self._control_allowed(info, "compressor"):
                self._apply_compressor(
                    ch,
                    info,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            if self._control_allowed(info, "pan"):
                self._apply_pan(
                    ch,
                    info,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            if self._control_allowed(info, "fader"):
                self._apply_gain_correction(ch, info, preset)
                self._apply_fader(
                    ch,
                    info,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            if self._control_allowed(info, "fx_send"):
                self._apply_fx_sends(
                    ch,
                    info,
                    preset,
                    phase_guard_context=phase_guard_context,
                )
            self._apply_group_bus_corrections(phase_guard_context=phase_guard_context)
            self._applied_channels.add(ch)
        except Exception as e:
            logger.error(f"Ch {ch}: single-channel correction error: {e}")

    def _handle_feedback_event(self, ch: int, event: FeedbackEvent):
        """React to a feedback event by applying notch or reducing fader."""
        info = self.channels.get(ch)
        if info is None:
            return
        if self._suppress_repeated_feedback_event(ch, event):
            return

        logger.warning(
            f"FEEDBACK Ch {ch}: {event.action} at {event.frequency_hz:.0f}Hz "
            f"({event.magnitude_db:.1f}dB)"
        )
        if event.action == "notch" and hasattr(self.mixer_client, "set_eq_band"):
            if not self._control_allowed(info, "feedback_notch"):
                self._log_processing_skip(ch, info, "feedback notch")
                return
            try:
                notch_band = self._select_feedback_notch_band(info)
                decision = self._execute_action(
                    EmergencyFeedbackNotch(
                        channel_id=ch,
                        band=notch_band,
                        freq_hz=event.frequency_hz,
                        q=10.0,
                        gain_db=-6.0,
                        reason="Emergency feedback notch",
                    ),
                    runtime_state=RuntimeState.EMERGENCY_FEEDBACK,
                )
                if decision and decision.sent:
                    self._record_feedback_action_sent(ch)
            except Exception:
                pass
        elif event.action == "fader_reduce":
            if not (
                self._control_allowed(info, "emergency_fader")
                or self._control_allowed(info, "fader")
            ):
                self._log_processing_skip(ch, info, "feedback fader reduction")
                return
            try:
                current = info.fader_db
                new_fader = max(-144.0, current - 3.0)
                decision = self._execute_action(
                    ChannelFaderMove(
                        channel_id=ch,
                        target_db=new_fader,
                        is_lead=info.source_role == "lead_vocal",
                        reason="Emergency feedback fader reduction",
                    ),
                    runtime_state=RuntimeState.EMERGENCY_FEEDBACK,
                )
                if decision and decision.sent:
                    actual_fader = getattr(decision.action, "target_db", new_fader)
                    info.fader_db = actual_fader
                    self._record_feedback_action_sent(ch)
                    logger.warning(f"FEEDBACK: Ch {ch} fader reduced to {actual_fader:.1f}dB")
            except Exception:
                pass

    def _suppress_repeated_feedback_event(self, ch: int, event: FeedbackEvent) -> bool:
        if self._feedback_actions_sent_total >= max(0, self.feedback_max_actions_per_run):
            logger.info(
                "Feedback event suppressed: run cap reached (%s actions)",
                self._feedback_actions_sent_total,
            )
            return True
        channel_count = self._feedback_actions_sent_by_channel.get(int(ch), 0)
        if channel_count >= max(0, self.feedback_max_actions_per_channel_per_run):
            logger.info(
                "Feedback event suppressed: channel %s cap reached (%s actions)",
                ch,
                channel_count,
            )
            return True

        bucket_hz = max(1.0, self.feedback_frequency_bucket_hz)
        bucket = int(round(float(event.frequency_hz or 0.0) / bucket_hz))
        key = (int(ch), str(event.action), bucket)
        now = time.monotonic()
        last = self._last_feedback_event_at.get(key)
        if (
            last is not None
            and (now - last) < max(0.0, self.feedback_event_min_interval_sec)
        ):
            logger.debug(
                "Feedback event suppressed by cooldown: ch=%s action=%s freq=%.0fHz",
                ch,
                event.action,
                event.frequency_hz,
            )
            return True
        channel_last = self._last_feedback_channel_action_at.get(int(ch))
        if (
            channel_last is not None
            and (now - channel_last) < max(0.0, self.feedback_channel_min_interval_sec)
        ):
            logger.debug(
                "Feedback event suppressed by channel cooldown: ch=%s action=%s freq=%.0fHz",
                ch,
                event.action,
                event.frequency_hz,
            )
            return True
        if (
            self._last_feedback_global_action_at > 0.0
            and (now - self._last_feedback_global_action_at)
            < max(0.0, self.feedback_global_min_interval_sec)
        ):
            logger.debug(
                "Feedback event suppressed by global cooldown: ch=%s action=%s freq=%.0fHz",
                ch,
                event.action,
                event.frequency_hz,
            )
            return True
        self._last_feedback_event_at[key] = now
        return False

    def _record_feedback_action_sent(self, ch: int):
        now = time.monotonic()
        channel = int(ch)
        self._last_feedback_channel_action_at[channel] = now
        self._last_feedback_global_action_at = now
        self._feedback_actions_sent_by_channel[channel] = (
            self._feedback_actions_sent_by_channel.get(channel, 0) + 1
        )
        self._feedback_actions_sent_total += 1

    def _select_feedback_notch_band(self, info: ChannelInfo) -> int:
        raw_settings = self._snapshot_raw(info)
        existing_bands = raw_settings.get("eq_bands") or []
        if not self.preserve_existing_processing or not existing_bands:
            return 4

        candidates = []
        for idx, band in enumerate(existing_bands[:4], start=1):
            gain = 0.0
            if len(band) >= 2:
                gain = self._as_float(band[1], 0.0)
            candidates.append((abs(gain), idx))

        if not candidates:
            return 4
        return min(candidates)[1]

    # ── Main run ─────────────────────────────────────────────────

    def run(self):
        """Run the full auto-soundcheck pipeline (blocking).

        Pipeline:
        1. Discover mixer on network (OSC/MIDI scan)
        2. Connect to mixer
        3. Scan audio devices, select best multichannel input
        4. Start audio capture
        5. Read channel names → recognize instruments
        6. Read existing channel settings (operator snapshot)
        7. Preserve current processing unless destructive reset is explicitly enabled
        8. Wait for audio signals
        9. Analyze signals (LUFS, peak, spectrum)
        10. Apply bounded corrections relative to the snapshot
        11. Enter monitoring mode (feedback detection, new channels)
        """
        self._stop_event.clear()
        self._phase_learning_snapshots = {}
        self._last_feedback_event_at = {}
        self._last_feedback_channel_action_at = {}
        self._last_feedback_global_action_at = 0.0
        self._feedback_actions_sent_by_channel = {}
        self._feedback_actions_sent_total = 0

        # Steps 1-2: Discover and connect mixer
        if not self._connect_mixer():
            self._set_state(EngineState.ERROR, "Mixer connection failed")
            return False

        # Steps 3-4: Scan and start audio
        if not self._start_audio():
            self._set_state(EngineState.ERROR, "Audio capture failed")
            return False

        if self.soundcheck_profile_enabled and self.soundcheck_profile_auto_load:
            self._load_soundcheck_profile()

        # Step 5: Scan channel names and recognize instruments
        self._scan_channels()

        # Step 6: Read existing settings from mixer
        self._read_channel_state()

        # Step 7: Preserve channel processing before analysis, unless explicitly opted into reset
        self._reset_channels()

        # Steps 8-9: Wait for audio signals and analyze
        self._wait_and_analyze()

        # Step 10: Apply bounded corrections over the preserved console state
        if self.use_decision_engine_v2:
            self._run_decision_engine_v2()
        elif self.auto_apply:
            self._apply_corrections()
        elif self.soundcheck_profile_capture_multiphase_learning:
            self._capture_learning_phase(
                "SNAPSHOT_LOCK",
                RuntimeState.SNAPSHOT_LOCK,
                metadata={
                    "applied_channel_ids": [],
                    "auto_apply": False,
                },
                notes="Snapshot lock captured without auto-apply",
            )

        if self.soundcheck_profile_enabled and self.soundcheck_profile_auto_save:
            profile = self._build_soundcheck_profile_from_live_buffers()
            if profile is not None:
                self._save_soundcheck_profile(profile)

        self._set_state(EngineState.RUNNING, "Auto-soundcheck complete, monitoring...")

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        try:
            while not self._stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping...")

        self.stop()
        return True

    def start_async(self):
        """Start engine in background thread."""
        self._engine_thread = threading.Thread(target=self.run, daemon=True)
        self._engine_thread.start()

    def restore_original_settings(self):
        """Restore channels to their original fader positions (before reset).

        This is a safety fallback: if the auto-soundcheck results are
        unacceptable, calling this method restores the fader levels
        that were captured before the reset step.
        """
        if not self.mixer_client:
            logger.warning("No mixer client — cannot restore settings")
            return

        restored = 0
        for ch, info in self.channels.items():
            snap = info.original_snapshot
            if snap is None or not snap.had_processing:
                continue
            try:
                if snap.fader_db > -90.0:
                    self.mixer_client.set_fader(ch, snap.fader_db)
                if snap.muted:
                    self.mixer_client.set_mute(ch, True)
                restored += 1
                logger.info(f"Ch {ch} '{info.name}': restored fader={snap.fader_db:.1f}dB")
            except Exception as e:
                logger.warning(f"Ch {ch}: restore failed: {e}")

        logger.info(f"Restored original settings for {restored} channel(s)")

    def stop(self):
        """Stop the engine, joining threads before releasing resources."""
        self._stop_event.set()
        self._set_state(EngineState.STOPPED, "Engine stopped")

        if self._observation_mixer_client is not None:
            summary = self._observation_mixer_client.get_summary()
            self._emit_observation(
                message=(
                    f"[OBSERVE] Engine complete. Intercepted "
                    f"{summary['total_operations']} mixer write operation(s)."
                ),
                summary=summary,
            )

        # Join monitor thread first so it stops accessing audio/mixer
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)

        self._evaluate_pending_actions(force=True)

        if self.audio_capture:
            try:
                self.audio_capture.stop()
            except Exception:
                pass

        if self.perceptual_evaluator is not None:
            try:
                self.perceptual_evaluator.stop()
            except Exception:
                pass

        if self.mixer_client:
            try:
                self.mixer_client.disconnect()
            except Exception:
                pass
        if self._real_mixer_client is not None:
            self.mixer_client = self._real_mixer_client
        self._observation_mixer_client = None
        self._real_mixer_client = None
        if self.autofoh_logger is not None:
            self.autofoh_logger.stop()
        self._generate_autofoh_session_report()

    def get_status(self) -> Dict:
        """Get engine status."""
        channels_info = {}
        for ch, info in self.channels.items():
            group_route = self.channel_group_routes.get(ch, {})
            channels_info[ch] = {
                "name": info.name,
                "preset": info.preset,
                "source_role": info.source_role,
                "stem_roles": list(info.stem_roles),
                "allowed_controls": list(info.allowed_controls),
                "classification_confidence": round(info.classification_confidence, 2),
                "classification_match_type": info.classification_match_type,
                "auto_corrections_enabled": info.auto_corrections_enabled,
                "recognized": info.recognized,
                "has_signal": info.has_signal,
                "peak_db": round(info.peak_db, 1),
                "rms_db": round(info.rms_db, 1),
                "lufs": round(info.lufs, 1),
                "gain_correction_db": round(info.gain_correction_db, 1),
                "eq_applied": info.eq_applied,
                "fader_db": round(info.fader_db, 1),
                "active_bus_sends": list(group_route.get("bus_ids", [])),
                "dca_assignments": list(group_route.get("dca_ids", [])),
            }

        discovery_info = None
        if self.discovered_mixer:
            discovery_info = {
                "mixer_type": self.discovered_mixer.mixer_type,
                "ip": self.discovered_mixer.ip,
                "port": self.discovered_mixer.port,
                "name": self.discovered_mixer.name,
                "method": self.discovered_mixer.discovery_method,
                "response_ms": round(self.discovered_mixer.response_time_ms, 0),
            }

        audio_device_info = None
        if self.selected_audio_device:
            audio_device_info = {
                "index": self.selected_audio_device.index,
                "name": self.selected_audio_device.name,
                "channels": self.selected_audio_device.max_input_channels,
                "protocol": self.selected_audio_device.protocol.value,
                "samplerate": self.selected_audio_device.default_samplerate,
                "score": self.selected_audio_device.score,
            }

        audio_devices_list = [
            {
                "index": d.index,
                "name": d.name,
                "channels": d.max_input_channels,
                "protocol": d.protocol.value,
                "is_multichannel": d.is_multichannel,
                "score": d.score,
            }
            for d in self.audio_devices
        ]

        return {
            "state": self.state.value,
            "runtime_state": self.runtime_state.value,
            "mixer_type": self.mixer_type,
            "mixer_ip": self.mixer_ip,
            "mixer_connected": self.mixer_client.is_connected if self.mixer_client else False,
            "audio_running": self.audio_capture.running if self.audio_capture else False,
            "audio_device": audio_device_info,
            "audio_devices_available": audio_devices_list,
            "total_channels": len(self.channels) if self.channels else self._configured_channel_count(),
            "selected_channels": self._iter_channels(),
            "master_reference_channels": sorted(self.master_reference_channels),
            "group_buses": {
                bus_id: {
                    "name": snapshot.name,
                    "fader_db": round(snapshot.fader_db, 1),
                    "muted": snapshot.muted,
                    "compressor_enabled": snapshot.compressor_enabled,
                    "dca_assignments": list(snapshot.dca_assignments),
                    "source_channels": list(snapshot.source_channels),
                }
                for bus_id, snapshot in sorted(self.bus_snapshots.items())
            },
            "dca_groups": {
                dca_id: {
                    "name": snapshot.name,
                    "fader_db": round(snapshot.fader_db, 1),
                    "muted": snapshot.muted,
                    "source_channels": list(snapshot.source_channels),
                    "source_buses": list(snapshot.source_buses),
                }
                for dca_id, snapshot in sorted(self.dca_snapshots.items())
            },
            "observe_only": self.observe_only,
            "decision_engine_v2": {
                "enabled": bool(self.use_decision_engine_v2),
                "dry_run": bool(self.decision_engine_v2_dry_run),
                "last_result": self.decision_engine_v2_last_result,
            },
            "applied_channel_ids": sorted(self._applied_channels),
            "safety_action_history_size": len(self.safety_controller.history) if self.safety_controller else 0,
            "pending_action_evaluations": len(self._pending_action_evaluations),
            "perceptual_enabled": self._perceptual_shadow_enabled(),
            "perceptual_mode": str(self.perceptual_config.get("mode", "shadow")),
            "perceptual_backend": (
                getattr(self.perceptual_evaluator.backend, "name", "")
                if self.perceptual_evaluator is not None
                else ""
            ),
            "perceptual_log_path": (
                str(getattr(self.perceptual_evaluator, "log_path", ""))
                if self._perceptual_shadow_enabled()
                else ""
            ),
            "muq_eval_enabled": self._muq_eval_enabled(),
            "muq_eval_shadow_mode": self._muq_shadow_mode(),
            "muq_eval_model_status": (
                getattr(self.muq_eval_service, "model_status", "")
                if self.muq_eval_service is not None
                else ""
            ),
            "muq_eval_log_path": str(self.muq_eval_config.get("log_path", "")),
            "muq_stem_drift": (
                self._last_muq_stem_drift
                if self._last_muq_stem_drift
                else (
                    self.muq_eval_service.stem_drift_status()
                    if self.muq_eval_service is not None
                    and hasattr(self.muq_eval_service, "stem_drift_status")
                    else {}
                )
            ),
            "shared_chat_mix_enabled": self.shared_chat_mix_config.enabled,
            "autofoh_log_enabled": self.autofoh_logging_enabled,
            "autofoh_log_path": str(self._default_autofoh_log_path()) if self.autofoh_logging_enabled else "",
            "autofoh_report_path": (
                str(self._default_autofoh_report_path())
                if self.autofoh_logging_enabled and self.autofoh_write_session_report_on_stop
                else ""
            ),
            "autofoh_session_report": (
                self.autofoh_session_report.to_dict()
                if self.autofoh_session_report is not None
                else {}
            ),
            "autofoh_session_report_summary": self.autofoh_session_report_summary,
            "autofoh_log_stats": (
                self.autofoh_logger.stats.__dict__
                if self.autofoh_logger is not None
                else {}
            ),
            "soundcheck_profile_enabled": self.soundcheck_profile_enabled,
            "soundcheck_profile_path": str(self._default_soundcheck_profile_path()) if self.soundcheck_profile_enabled else "",
            "loaded_soundcheck_profile_name": (
                self.loaded_soundcheck_profile.name
                if self.loaded_soundcheck_profile is not None
                else ""
            ),
            "soundcheck_learning_phases": sorted(self._phase_learning_snapshots.keys()),
            "loaded_phase_target_names": (
                sorted(self.loaded_soundcheck_profile.phase_targets.keys())
                if self.loaded_soundcheck_profile is not None
                else []
            ),
            "current_target_corridor": self.current_target_corridor.to_dict(),
            "channels_with_signal": sum(1 for c in self.channels.values() if c.has_signal),
            "channels_recognized": sum(1 for c in self.channels.values() if c.recognized),
            "discovered": discovery_info,
            "channels": channels_info,
        }


# ── CLI entry point ──────────────────────────────────────────────

def main():
    """CLI entry point for headless auto-soundcheck."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AUTO-MIXER Tubeslave — Automatic Soundcheck Engine"
    )
    parser.add_argument("--mixer", default=None, choices=["dlive", "wing"],
                        help="Mixer type (default: auto-detect)")
    parser.add_argument("--ip", default=None,
                        help="Mixer IP address (default: auto-discover)")
    parser.add_argument("--port", type=int, default=None,
                        help="Mixer port (default: auto)")
    parser.add_argument("--tls", action="store_true",
                        help="Use TLS for dLive connection")
    parser.add_argument("--no-discover", action="store_true",
                        help="Skip auto-discovery, use --ip directly")
    parser.add_argument("--full-scan", action="store_true",
                        help="Full /24 subnet scan (slower but thorough)")
    parser.add_argument("--audio-device", default=None,
                        help="Audio device name pattern (e.g. 'soundgrid', 'dante')")
    parser.add_argument("--channels", type=int, default=48,
                        help="Number of input channels (default: 48)")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Sample rate (default: 48000)")
    parser.add_argument("--no-apply", action="store_true",
                        help="Analyze only, do not apply corrections")
    parser.add_argument("--use-decision-engine-v2", action="store_true",
                        help="Use opt-in Analyzer -> Knowledge -> Critic -> Decision Engine -> Safety Gate -> Executor path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze and recommend only; do not send mixer writes")
    parser.add_argument("--offline-experiment", action="store_true",
                        help="Run the v2 offline experiment harness instead of connecting to a live mixer")
    parser.add_argument("--experiment-input", default="",
                        help="Metrics JSON for --offline-experiment")
    parser.add_argument("--experiment-output-dir", default="",
                        help="Output directory for --offline-experiment reports")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--scan-only", action="store_true",
                        help="Only scan for mixers, do not connect")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Scan-only mode
    if args.scan_only:
        from mixer_discovery import main as discovery_main
        discovery_main()
        return

    if args.offline_experiment:
        from automixer.experiments import ExperimentRunner

        if args.experiment_input:
            payload = json.loads(Path(args.experiment_input).expanduser().read_text(encoding="utf-8"))
        else:
            payload = {
                "analyzer_output": {
                    "source_module": "auto_soundcheck_engine_cli_sample",
                    "channels": [
                        {
                            "channel_id": 1,
                            "name": "Sample Lead Vocal",
                            "role": "lead_vocal",
                            "metrics": {
                                "lufs": -24.0,
                                "target_lufs": -20.0,
                                "true_peak_dbtp": -8.0,
                                "crest_factor_db": 14.0,
                                "SibilanceIndex": 3.0,
                            },
                            "confidence": 0.9,
                        }
                    ],
                },
                "current_state": {"channel:1": {"fader_db": -12.0, "true_peak_dbtp": -8.0}},
            }
        output_dir = args.experiment_output_dir or str(Path.cwd() / "decision_engine_v2_experiment")
        report = ExperimentRunner().run(payload, output_dir)
        print(json.dumps({"artifacts": report.get("artifacts", {})}, ensure_ascii=False))
        return

    auto_discover = not args.no_discover

    def on_state(state, msg):
        print(f"\n>>> [{state}] {msg}")

    def on_channel(ch, data):
        print(f"  Ch {ch}: {data['name']} -> {data.get('preset', '?')} | "
              f"peak={data['peak_db']:.1f}dB | lufs={data['lufs']:.1f} | "
              f"gain_corr={data['gain_correction_db']:+.1f}dB | "
              f"fader={data['fader_db']:.1f}dB | EQ={data['eq_applied']}")

    engine = AutoSoundcheckEngine(
        mixer_type=args.mixer,
        mixer_ip=args.ip,
        mixer_port=args.port,
        mixer_tls=args.tls,
        audio_device_name=args.audio_device,
        num_channels=args.channels,
        sample_rate=args.sample_rate,
        auto_apply=(not args.no_apply) and (not args.dry_run),
        auto_discover=auto_discover,
        scan_subnet=args.full_scan,
        on_state_change=on_state,
        on_channel_update=on_channel,
        use_decision_engine_v2=args.use_decision_engine_v2,
        decision_engine_dry_run=args.dry_run,
    )

    print("=" * 60)
    print("  AUTO-MIXER Tubeslave — Automatic Soundcheck")
    if args.ip:
        mixer_desc = f"{(args.mixer or 'auto').upper()} @ {args.ip}:{args.port or 'auto'}"
    else:
        mixer_desc = "Auto-discover on network"
    print(f"  Mixer: {mixer_desc}")
    print(f"  Audio: {args.audio_device or 'auto-detect'}")
    print(f"  Channels: {args.channels}")
    print(f"  Auto-apply: {(not args.no_apply) and (not args.dry_run)}")
    print(f"  Decision Engine v2: {args.use_decision_engine_v2}")
    print(f"  Dry-run: {args.dry_run}")
    print(f"  Auto-discover: {auto_discover}")
    print("=" * 60)
    print()

    engine.run()


if __name__ == "__main__":
    main()
