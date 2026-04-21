"""
Auto Soundcheck Engine — headless orchestrator for automatic mixing.

Connects to mixer (dLive or WING), detects audio device (SoundGrid/Dante),
reads channel names, recognizes instruments, waits for audio signals,
and automatically applies gain staging, EQ, compressor, and fader corrections
without user confirmation.

This is the main entry point for fully automatic operation.
"""

import asyncio
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
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    CompressorAdjust,
    EmergencyFeedbackNotch,
    HighPassAdjust,
    PanAdjust,
    SafetyDecision,
    SendLevelAdjust,
)
from channel_recognizer import (
    classification_from_legacy_preset,
    recognize_instrument, recognize_instrument_spectral_fallback,
    scan_and_recognize, AVAILABLE_PRESETS,
)
from config_manager import ConfigManager
from feedback_detector import FeedbackDetector, FeedbackEvent
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
    """Backup of a channel's settings before reset."""
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

        default_config_path = Path(config_path) if config_path else (
            Path(__file__).resolve().parents[1] / "config" / "automixer.yaml"
        )
        resolved_config_path = str(default_config_path) if default_config_path.exists() else config_path
        self.config_manager = ConfigManager(config_path=resolved_config_path)
        autofoh_config = self.config_manager.get_section("autofoh")
        self.classifier_config = autofoh_config.get("classifier", {})
        autofoh_safety = autofoh_config.get("safety", {})
        autofoh_evaluation = autofoh_config.get("evaluation", {})
        autofoh_logging = autofoh_config.get("logging", {})
        autofoh_soundcheck_profile = autofoh_config.get("soundcheck_profile", {})
        self.autofoh_analysis_config = autofoh_config.get("analysis", {})
        detector_config = autofoh_config.get("detectors", {})
        self.monitor_cycle_interval_sec = float(
            detector_config.get("monitor_cycle_interval_sec", 1.0)
        )
        self.minimum_auto_apply_classification_confidence = float(
            autofoh_safety.get("minimum_auto_apply_classification_confidence", 0.75)
        )
        self.new_or_unknown_channel_auto_corrections_enabled = bool(
            autofoh_safety.get("new_or_unknown_channel_auto_corrections_enabled", False)
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
        self.safety_controller: Optional[AutoFOHSafetyController] = None
        self.autofoh_logger: Optional[AutoFOHStructuredLogger] = None
        self.autofoh_session_report: Optional[AutoFOHSessionReport] = None
        self.autofoh_session_report_summary: str = ""
        self.loaded_soundcheck_profile: Optional[AutoFOHSoundcheckProfile] = None
        self.discovered_mixer: Optional[DiscoveredMixer] = None
        self.selected_audio_device: Optional[AudioDevice] = None
        self.audio_devices: List[AudioDevice] = []

        self.state = EngineState.IDLE
        self.runtime_state = RuntimeState.IDLE
        self.channels: Dict[int, ChannelInfo] = {}
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
            f"eval_enabled={self.evaluation_policy.enabled}, "
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
        if not self.evaluation_policy.enabled:
            return None
        channel_id = getattr(action, "channel_id", None)
        if channel_id is None:
            return None
        snapshot = snapshot or {}
        self._action_evaluation_seq += 1
        pending = PendingActionEvaluation(
            evaluation_id=self._action_evaluation_seq,
            action=action,
            registered_at=time.monotonic(),
            due_at=time.monotonic() + self.evaluation_policy.evaluation_window_sec,
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
        return True, {"note": "control-state verification not implemented for action type"}

    def _evaluate_pending_actions(self, force: bool = False) -> List[ActionEvaluationOutcome]:
        if not self.evaluation_policy.enabled or not self._pending_action_evaluations:
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

            if outcome.should_rollback and outcome.rollback_action is not None:
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
            self._register_pending_action_evaluation(decision.action, effective_state, snapshot)
        return decision

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

    def _control_allowed(self, info: ChannelInfo, control_name: str) -> bool:
        if control_name in {"feedback_notch", "emergency_fader"}:
            return control_name in info.allowed_controls
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

    def _read_channel_state(self):
        """Read current processing settings for all channels.

        Determines which channels already have non-default settings
        (EQ, HPF, fader, gain) so we can back them up before resetting.
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

            fader_db = raw_settings.get("fader_db", -100.0)
            muted = raw_settings.get("muted", False)

            # Detect if channel has non-default processing
            has_processing = False
            if fader_db > -90.0:
                has_processing = True
            if not muted and fader_db > -50.0:
                has_processing = True

            snapshot = ChannelSnapshot(
                channel=ch,
                fader_db=fader_db,
                muted=muted,
                had_processing=has_processing,
                raw_settings=raw_settings,
            )
            info.original_snapshot = snapshot

            if has_processing:
                channels_with_processing += 1
                logger.debug(
                    f"Ch {ch} '{info.name}': existing settings detected "
                    f"(fader={fader_db:.1f}dB, muted={muted})"
                )

        logger.info(
            f"Channel state read: {channels_with_processing}/{len(self.channels)} "
            f"channels have existing settings"
        )

    # ── 3c. Reset channels to neutral before analysis ────────────

    def _reset_channels(self):
        """Reset all channel processing to flat/neutral.

        This ensures that the audio we analyze is the raw unprocessed
        signal from the stage. After analysis, new settings will be
        applied from scratch.
        """
        self._set_state(
            EngineState.RESETTING,
            "Resetting channel processing to neutral..."
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
        if isinstance(action, ChannelEQMove):
            return "eq"
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
                if action.gain_db < 0.0 and band_error_db <= green_delta_db:
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

        Channels have been reset to neutral before analysis, so all
        settings are applied from scratch to clean channels.
        """
        self._set_state(EngineState.APPLYING, "Applying full processing chain...")
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

            preset = info.preset or "custom"
            had_proc = ""
            if info.original_snapshot and info.original_snapshot.had_processing:
                had_proc = " (previous settings cleared)"
            logger.info(
                f"Ch {ch} '{info.name}' (preset={preset}): "
                f"applying full processing chain{had_proc}..."
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

        # 9. Phase / polarity check across correlated channel pairs
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

        try:
            if hasattr(self.mixer_client, 'set_hpf'):
                decision = self._execute_action(
                    HighPassAdjust(
                        channel_id=ch,
                        freq_hz=adaptive_freq,
                        enabled=True,
                        reason=f"HPF preset for {preset}",
                    ),
                    phase_guard_context=phase_guard_context,
                )
                if decision and decision.sent and abs(adaptive_freq - base_freq) > 5:
                    logger.info(f"Ch {ch}: HPF={adaptive_freq:.0f}Hz (preset {base_freq:.0f}Hz)")
                elif decision and decision.sent:
                    logger.debug(f"Ch {ch}: HPF={adaptive_freq:.0f}Hz")
        except Exception as e:
            logger.warning(f"Ch {ch}: HPF failed: {e}")

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

        try:
            sent_any = False
            for band_idx, (freq, gain, q) in enumerate(adapted_bands, start=1):
                if band_idx > 4:
                    break
                decision = self._execute_action(
                    ChannelEQMove(
                        channel_id=ch,
                        band=band_idx,
                        freq_hz=freq,
                        gain_db=gain,
                        q=q,
                        reason=f"EQ preset for {preset}",
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

        # Start with base + LUFS gain correction (from _apply_gain_correction)
        fader_db = base_fader + info.gain_correction_db

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

        fader_db = max(-30.0, min(0.0, fader_db))
        previous_fader_db = float(info.fader_db)

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
                info.fader_db = fader_db
            else:
                info.fader_db = previous_fader_db

            total_adj = fader_db - base_fader
            if decision and decision.sent and abs(total_adj) > 0.5:
                logger.info(
                    f"Ch {ch} '{info.name}': fader={fader_db:.1f}dB "
                    f"(base={base_fader:.1f}, lufs_corr={info.gain_correction_db:+.1f}, "
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

        try:
            decision = self._execute_action(
                ChannelGainMove(
                    channel_id=ch,
                    target_db=correction,
                    reason=f"Input gain staging for {preset}",
                ),
                phase_guard_context=phase_guard_context,
            )
            if decision and decision.sent:
                logger.info(
                    f"Ch {ch} '{info.name}': input gain {correction:+.1f}dB "
                    f"(TP={true_peak:.1f}dBTP, RMS={rms:.1f}, "
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
                    logger.warning(
                        f"Phase: {name_b} is inverted (corr={icm.cross_correlation:.3f}), "
                        f"flipping polarity"
                    )
                    if hasattr(self.mixer_client, 'set_polarity'):
                        self.mixer_client.set_polarity(ch_b, True)

                # Time alignment
                elif icm.delay_ms > 0.1 and icm.coherence > 0.3:
                    align_ch = ch_b if icm.delay_samples > 0 else ch_a
                    logger.info(
                        f"Phase: applying {icm.delay_ms:.2f}ms delay "
                        f"to Ch {align_ch} (coherence={icm.coherence:.3f})"
                    )
                    if hasattr(self.mixer_client, 'set_delay'):
                        self.mixer_client.set_delay(align_ch, icm.delay_ms, enabled=True)

            except Exception as e:
                logger.warning(f"Phase check error for pair ({ch_a}, {ch_b}): {e}")

    def _find_correlated_pairs(self) -> List[Tuple[int, int, str]]:
        """Find channel pairs that should be phase-correlated.

        Looks for adjacent channels with the same preset type
        (e.g. two 'overheads', two 'synth', etc.).
        """
        pairs = []
        preset_channels: Dict[str, List[int]] = {}

        for ch, info in self.channels.items():
            if info.has_signal and info.preset:
                preset_channels.setdefault(info.preset, []).append(ch)

        for preset, channels in preset_channels.items():
            if len(channels) < 2:
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

    # ── 5g. Compressor ───────────────────────────────────────────

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

        try:
            decision = self._execute_action(
                CompressorAdjust(
                    channel_id=ch,
                    threshold_db=round(threshold, 1),
                    ratio=round(ratio, 1),
                    attack_ms=round(attack, 1),
                    release_ms=round(release, 1),
                    makeup_db=0.0,
                    enabled=True,
                    reason=f"Compressor profile for {preset}",
                ),
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

            adapted = f" (adapted: {', '.join(changes)})" if changes else ""
            if decision and decision.sent:
                logger.info(
                    f"Ch {ch} '{info.name}': compressor thr={threshold:.0f}dB "
                    f"ratio={ratio:.1f}:1 atk={attack:.0f}ms rel={release:.0f}ms{adapted}"
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

    # ── 6. Continuous monitoring ─────────────────────────────────

    def _monitor_loop(self):
        """Continuous monitoring: feedback detection + cautious mix cleanup."""
        self.feedback_detector = FeedbackDetector(
            sample_rate=self.sample_rate,
            fft_size=2048,
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
        """Apply full processing chain for a single newly detected channel.

        Resets the channel, runs a short analysis pass for fresh metrics,
        then applies the full processing chain.
        """
        info = self.channels.get(ch)
        if info is None or ch in self._applied_channels:
            return
        if not info.auto_corrections_enabled:
            self._log_processing_skip(ch, info, "single-channel auto-processing")
            return
        preset = info.preset or "custom"
        try:
            self.mixer_client.reset_channel_processing(ch)
            info.was_reset = True
            time.sleep(0.3)

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
            self._applied_channels.add(ch)
        except Exception as e:
            logger.error(f"Ch {ch}: single-channel correction error: {e}")

    def _handle_feedback_event(self, ch: int, event: FeedbackEvent):
        """React to a feedback event by applying notch or reducing fader."""
        logger.warning(
            f"FEEDBACK Ch {ch}: {event.action} at {event.frequency_hz:.0f}Hz "
            f"({event.magnitude_db:.1f}dB)"
        )
        info = self.channels.get(ch)
        if info is None:
            return
        if event.action == "notch" and hasattr(self.mixer_client, "set_eq_band"):
            if not self._control_allowed(info, "feedback_notch"):
                self._log_processing_skip(ch, info, "feedback notch")
                return
            try:
                self._execute_action(
                    EmergencyFeedbackNotch(
                        channel_id=ch,
                        band=4,
                        freq_hz=event.frequency_hz,
                        q=10.0,
                        gain_db=-6.0,
                        reason="Emergency feedback notch",
                    ),
                    runtime_state=RuntimeState.EMERGENCY_FEEDBACK,
                )
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
                new_fader = max(-30.0, current - 3.0)
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
                    info.fader_db = new_fader
                    logger.warning(f"FEEDBACK: Ch {ch} fader reduced to {new_fader:.1f}dB")
            except Exception:
                pass

    # ── Main run ─────────────────────────────────────────────────

    def run(self):
        """Run the full auto-soundcheck pipeline (blocking).

        Pipeline:
        1. Discover mixer on network (OSC/MIDI scan)
        2. Connect to mixer
        3. Scan audio devices, select best multichannel input
        4. Start audio capture
        5. Read channel names → recognize instruments
        6. Read existing channel settings (backup)
        7. Reset all channels to neutral (flat EQ, HPF off)
        8. Wait for audio signals (now receiving clean/raw audio)
        9. Analyze signals (LUFS, peak, spectrum)
        10. Apply new corrections (HPF, EQ, gain, fader)
        11. Enter monitoring mode (feedback detection, new channels)
        """
        self._stop_event.clear()
        self._phase_learning_snapshots = {}

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

        # Step 7: Reset all channels to neutral before analysis
        self._reset_channels()

        # Steps 8-9: Wait for clean audio signals and analyze
        self._wait_and_analyze()

        # Step 10: Apply new corrections from scratch
        if self.auto_apply:
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
            "observe_only": self.observe_only,
            "applied_channel_ids": sorted(self._applied_channels),
            "safety_action_history_size": len(self.safety_controller.history) if self.safety_controller else 0,
            "pending_action_evaluations": len(self._pending_action_evaluations),
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
        auto_apply=not args.no_apply,
        auto_discover=auto_discover,
        scan_subnet=args.full_scan,
        on_state_change=on_state,
        on_channel_update=on_channel,
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
    print(f"  Auto-apply: {not args.no_apply}")
    print(f"  Auto-discover: {auto_discover}")
    print("=" * 60)
    print()

    engine.run()


if __name__ == "__main__":
    main()
