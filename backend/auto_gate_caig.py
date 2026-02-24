"""
Auto Gate - Cross-Adaptive Intelligent Gate (CAIG)

Based on Kimi AI debate results:
- Cross-Adaptive Intelligent Gate architecture
- Feature extraction: RMS, Peak, Crest Factor, LF Energy (20-250Hz)
- Adaptive threshold with noise floor tracking
- Group-based processing (Drums, Bass, Vocals, Guitars, Keys)
- Drum kit priority rules

References:
- Intelligent Music Production (De Man et al., 2020)
- SAR (Signal to Artifact Ratio) optimization
- δb (reduction in bleed) maximization
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GateState(Enum):
    """Gate state machine."""
    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    HOLD = "hold"
    RELEASING = "releasing"


class InstrumentGroup(Enum):
    """Instrument groups for cross-adaptive processing."""
    DRUMS = "drums"
    BASS = "bass"
    VOCALS = "vocals"
    GUITARS = "guitars"
    KEYS = "keys"
    OTHER = "other"


@dataclass
class GateFeatures:
    """Features for gate decision."""
    rms_db: float = -100.0
    peak_db: float = -100.0
    crest_factor_db: float = 0.0
    lf_energy_db: float = -100.0  # 20-250Hz
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'rms_db': self.rms_db,
            'peak_db': self.peak_db,
            'crest_factor_db': self.crest_factor_db,
            'lf_energy_db': self.lf_energy_db
        }


@dataclass
class GateSettings:
    """Gate settings for a channel."""
    threshold_db: float = -60.0
    attack_ms: float = 0.5
    release_ms: float = 80.0
    hold_ms: float = 10.0
    range_db: float = -80.0  # Max attenuation when closed
    hysteresis_db: float = 3.0
    
    # Adaptive settings
    adaptive_threshold: bool = True
    noise_floor_db: float = -70.0
    noise_floor_margin_db: float = 6.0


@dataclass
class GateDecision:
    """Gate decision for a channel."""
    channel_id: int
    state: GateState
    gain_db: float = 0.0
    threshold_db: float = -60.0
    is_triggered: bool = False
    group_influence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'channel_id': self.channel_id,
            'state': self.state.value,
            'gain_db': self.gain_db,
            'threshold_db': self.threshold_db,
            'is_triggered': self.is_triggered,
            'group_influence': self.group_influence
        }


class FeatureExtractor:
    """
    Extract features for gate decision every 64 samples (1.33ms @ 48kHz).
    
    Features:
    - RMS: Signal energy
    - Peak: Max amplitude
    - Crest Factor: Peak/RMS ratio (transient detection)
    - LF Energy: 20-250Hz energy (kick detection)
    """
    
    def __init__(self, sample_rate: int = 48000, frame_size: int = 64):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration_ms = 1000.0 * frame_size / sample_rate
        
        # LF filter coefficients (20-250Hz bandpass)
        self.lf_low_cutoff = 20.0
        self.lf_high_cutoff = 250.0
        
        logger.info(f"FeatureExtractor initialized: {frame_size} samples ({self.frame_duration_ms:.2f}ms)")
    
    def extract(self, audio: np.ndarray) -> GateFeatures:
        """Extract features from audio frame."""
        features = GateFeatures()
        
        # Check for empty audio
        if audio.size == 0:
            return features
        
        # RMS
        rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
        features.rms_db = 20 * np.log10(rms)
        
        # Peak
        peak = np.max(np.abs(audio))
        features.peak_db = 20 * np.log10(peak + 1e-10)
        
        # Crest Factor
        if rms > 0:
            crest_linear = peak / rms
            features.crest_factor_db = 20 * np.log10(crest_linear + 1e-10)
        
        # LF Energy (simplified - use FFT in production)
        # For now, approximate with weighted low-frequency content
        lf_audio = self._simple_lpf(audio)
        lf_rms = np.sqrt(np.mean(lf_audio ** 2) + 1e-10)
        features.lf_energy_db = 20 * np.log10(lf_rms)
        
        return features
    
    def _simple_lpf(self, audio: np.ndarray) -> np.ndarray:
        """Simple low-pass filter for LF energy estimation."""
        # Simple moving average as LPF approximation
        window_size = max(1, int(self.sample_rate / 500))  # ~500Hz cutoff
        if len(audio) >= window_size:
            kernel = np.ones(window_size) / window_size
            return np.convolve(audio, kernel, mode='same')
        return audio


class GroupAnalyzer:
    """
    Analyze group features for cross-adaptive processing.
    
    Groups:
    - Drums (Kick, Snare, Toms, Overheads)
    - Bass
    - Vocals
    - Guitars
    - Keys
    """
    
    def __init__(self):
        self.group_channels: Dict[InstrumentGroup, List[int]] = {}
        self.group_features: Dict[InstrumentGroup, Dict[str, float]] = {}
        
        logger.info("GroupAnalyzer initialized")
    
    def assign_channel(self, channel_id: int, group: InstrumentGroup):
        """Assign channel to a group."""
        if group not in self.group_channels:
            self.group_channels[group] = []
        if channel_id not in self.group_channels[group]:
            self.group_channels[group].append(channel_id)
    
    def analyze_groups(self, all_features: Dict[int, GateFeatures]) -> Dict[InstrumentGroup, Dict[str, float]]:
        """
        Calculate group features.
        
        Returns:
            {group: {'avg_rms', 'max_peak', 'dominant_channel', 'activity'}}
        """
        group_data = {}
        
        for group, channels in self.group_channels.items():
            if not channels:
                continue
            
            # Collect features for group
            rms_values = []
            peak_values = []
            dominant_ch = None
            max_rms = -100
            
            for ch in channels:
                if ch in all_features:
                    feat = all_features[ch]
                    rms_values.append(feat.rms_db)
                    peak_values.append(feat.peak_db)
                    
                    if feat.rms_db > max_rms:
                        max_rms = feat.rms_db
                        dominant_ch = ch
            
            if rms_values:
                group_data[group] = {
                    'avg_rms': np.mean(rms_values),
                    'max_peak': max(peak_values) if peak_values else -100,
                    'dominant_channel': dominant_ch,
                    'activity': len([r for r in rms_values if r > -60]) / len(rms_values)
                }
        
        self.group_features = group_data
        return group_data
    
    def get_group_influence(self, channel_id: int, group: InstrumentGroup) -> float:
        """
        Calculate how much group activity should affect this channel.
        
        Returns:
            Influence in dB (0-9dB range)
        """
        if group not in self.group_features:
            return 0.0
        
        group_data = self.group_features[group]
        
        # If this channel is dominant, less influence
        if group_data['dominant_channel'] == channel_id:
            return 0.0
        
        # More active group = more influence
        activity = group_data['activity']
        influence = activity * 9.0  # Max 9dB offset
        
        return influence


class AdaptiveThreshold:
    """
    Adaptive threshold with noise floor tracking and cross-adaptive offset.
    
    Formula:
    threshold = noise_floor + 6dB + crest_offset + cross_adaptive_offset
    
    Where:
    - noise_floor: tracked minimum RMS
    - crest_offset: +6dB for transients, -3dB for sustain
    - cross_adaptive_offset: 0-9dB based on group activity
    """
    
    def __init__(self, tracking_coeff: float = 0.99):
        self.tracking_coeff = tracking_coeff
        self.noise_floor_db = -70.0
        
        logger.info("AdaptiveThreshold initialized")
    
    def update_noise_floor(self, rms_db: float):
        """Update noise floor with slow tracking."""
        # Only track when signal is low
        if rms_db < self.noise_floor_db + 10:
            self.noise_floor_db = (self.tracking_coeff * self.noise_floor_db + 
                                  (1 - self.tracking_coeff) * rms_db)
    
    def calculate_threshold(
        self,
        features: GateFeatures,
        group_influence: float = 0.0,
        base_margin_db: float = 6.0
    ) -> float:
        """
        Calculate adaptive threshold.
        
        Args:
            features: Current features
            group_influence: Cross-adaptive influence from group
            base_margin_db: Base margin above noise floor
            
        Returns:
            Threshold in dB
        """
        # Base threshold
        threshold = self.noise_floor_db + base_margin_db
        
        # Crest factor offset
        if features.crest_factor_db > 15:  # Transient
            crest_offset = 6.0
        elif features.crest_factor_db < 8:  # Sustain
            crest_offset = -3.0
        else:
            crest_offset = 0.0
        
        threshold += crest_offset
        
        # Cross-adaptive offset
        threshold += group_influence
        
        return threshold


class DrumKitRules:
    """
    Priority rules for drum kit processing.
    
    Priorities (1-4, 4 = highest):
    - Kick: 4
    - Snare: 3
    - Toms: 2
    - Overheads: 1
    
    Rules:
    1. Kick suppresses Snare (+6dB threshold during Kick)
    2. Overheads never fully closed (min gain -10dB)
    """
    
    PRIORITIES = {
        'kick': 4,
        'snare': 3,
        'tom': 2,
        'overheads': 1
    }
    
    def __init__(self):
        self.drum_channels: Dict[str, int] = {}  # instrument -> channel_id
        
        logger.info("DrumKitRules initialized")
    
    def register_drum(self, instrument: str, channel_id: int):
        """Register a drum channel."""
        self.drum_channels[instrument] = channel_id
    
    def apply_rules(
        self,
        channel_id: int,
        instrument: str,
        base_threshold: float,
        all_decisions: Dict[int, GateDecision]
    ) -> float:
        """
        Apply drum kit rules to threshold.
        
        Returns:
            Modified threshold
        """
        threshold = base_threshold
        
        # Rule 1: Kick suppresses Snare
        if instrument == 'snare':
            kick_ch = self.drum_channels.get('kick')
            if kick_ch and kick_ch in all_decisions:
                kick_decision = all_decisions[kick_ch]
                if kick_decision.is_triggered:
                    # Snare threshold +6dB during Kick
                    threshold += 6.0
        
        return threshold
    
    def get_min_gain(self, instrument: str) -> float:
        """Get minimum gain for instrument (overheads never fully closed)."""
        if instrument == 'overheads':
            return -10.0  # Never fully closed
        return -80.0  # Full range


class GateProcessor:
    """
    Main gate processor with state machine.
    
    States: CLOSED → OPENING → OPEN → HOLD → RELEASING → CLOSED
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # State
        self.state = GateState.CLOSED
        self.current_gain_db = -80.0
        self.hold_counter = 0
        
        # Timing
        self.attack_samples = 0
        self.release_samples = 0
        self.hold_samples = 0
        
        logger.info("GateProcessor initialized")
    
    def configure(self, settings: GateSettings):
        """Configure gate timing."""
        self.attack_samples = int(settings.attack_ms * self.sample_rate / 1000)
        self.release_samples = int(settings.release_ms * self.sample_rate / 1000)
        self.hold_samples = int(settings.hold_ms * self.sample_rate / 1000)
    
    def process(
        self,
        features: GateFeatures,
        threshold_db: float,
        settings: GateSettings
    ) -> GateDecision:
        """Process one frame and return decision."""
        decision = GateDecision(channel_id=0, state=self.state)
        
        # Check if signal is above threshold
        is_above = features.rms_db > threshold_db
        
        # State machine
        if self.state == GateState.CLOSED:
            if is_above:
                self.state = GateState.OPENING
                self.current_gain_db = settings.range_db
        
        elif self.state == GateState.OPENING:
            # Attack phase
            gain_step = abs(settings.range_db) / max(1, self.attack_samples)
            self.current_gain_db += gain_step
            
            if self.current_gain_db >= 0:
                self.current_gain_db = 0
                self.state = GateState.OPEN
        
        elif self.state == GateState.OPEN:
            if not is_above:
                self.state = GateState.HOLD
                self.hold_counter = self.hold_samples
        
        elif self.state == GateState.HOLD:
            self.hold_counter -= 1
            if self.hold_counter <= 0:
                self.state = GateState.RELEASING
        
        elif self.state == GateState.RELEASING:
            # Release phase
            gain_step = abs(settings.range_db) / max(1, self.release_samples)
            self.current_gain_db -= gain_step
            
            if self.current_gain_db <= settings.range_db:
                self.current_gain_db = settings.range_db
                self.state = GateState.CLOSED
        
        # Fill decision
        decision.state = self.state
        decision.gain_db = self.current_gain_db
        decision.threshold_db = threshold_db
        decision.is_triggered = (self.state in [GateState.OPENING, GateState.OPEN, GateState.HOLD])
        
        return decision


class AutoGateController:
    """
    Main controller for Cross-Adaptive Intelligent Gate (CAIG).
    
    Integrates all components:
    1. FeatureExtractor (64 samples frame)
    2. GroupAnalyzer (cross-adaptive)
    3. AdaptiveThreshold (noise floor tracking)
    4. DrumKitRules (priority processing)
    5. GateProcessor (state machine)
    """
    
    def __init__(
        self,
        num_channels: int = 32,
        sample_rate: int = 48000,
        mixer_client=None
    ):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.mixer_client = mixer_client
        
        # Components
        self.extractor = FeatureExtractor(sample_rate)
        self.group_analyzer = GroupAnalyzer()
        self.threshold_calculator = AdaptiveThreshold()
        self.drum_rules = DrumKitRules()
        
        # Per-channel processors
        self.processors: Dict[int, GateProcessor] = {}
        self.settings: Dict[int, GateSettings] = {}
        self.instruments: Dict[int, str] = {}
        self.groups: Dict[int, InstrumentGroup] = {}
        
        # Initialize processors
        for ch in range(num_channels):
            self.processors[ch] = GateProcessor(sample_rate)
            self.settings[ch] = GateSettings()
        
        # Default drum assignments
        self._setup_default_drums()
        
        logger.info(f"AutoGateController initialized: {num_channels} channels")
    
    def _setup_default_drums(self):
        """Setup default drum channel assignments."""
        # These would be configured by user
        pass
    
    def configure_channel(
        self,
        channel_id: int,
        instrument: str,
        group: InstrumentGroup,
        attack_ms: float = 0.5,
        release_ms: float = 80.0,
        is_drum: bool = False
    ):
        """Configure a channel."""
        self.instruments[channel_id] = instrument
        self.groups[channel_id] = group
        
        # Register drum
        if is_drum:
            self.drum_rules.register_drum(instrument, channel_id)
        
        # Configure settings
        settings = GateSettings()
        settings.attack_ms = attack_ms
        settings.release_ms = release_ms
        self.settings[channel_id] = settings
        
        # Configure processor
        self.processors[channel_id].configure(settings)
        
        # Assign to group
        self.group_analyzer.assign_channel(channel_id, group)
    
    def process(self, audio_data: Dict[int, np.ndarray]) -> Dict[int, GateDecision]:
        """
        Process audio and return gate decisions.
        
        Args:
            audio_data: {channel_id: audio_samples}
            
        Returns:
            {channel_id: GateDecision}
        """
        # Step 1: Extract features
        features = {}
        for ch, audio in audio_data.items():
            if ch < self.num_channels:
                features[ch] = self.extractor.extract(audio)
                
                # Update noise floor
                self.threshold_calculator.update_noise_floor(features[ch].rms_db)
        
        # Step 2: Analyze groups
        group_data = self.group_analyzer.analyze_groups(features)
        
        # Step 3: Process each channel
        decisions = {}
        for ch, audio in audio_data.items():
            if ch >= self.num_channels:
                continue
            
            feat = features[ch]
            settings = self.settings[ch]
            
            # Calculate base threshold
            group = self.groups.get(ch, InstrumentGroup.OTHER)
            group_influence = self.group_analyzer.get_group_influence(ch, group)
            
            threshold = self.threshold_calculator.calculate_threshold(
                feat, group_influence
            )
            
            # Apply drum kit rules
            instrument = self.instruments.get(ch, 'unknown')
            threshold = self.drum_rules.apply_rules(
                ch, instrument, threshold, decisions
            )
            
            # Process gate
            decision = self.processors[ch].process(feat, threshold, settings)
            decision.channel_id = ch
            decision.group_influence = group_influence
            
            decisions[ch] = decision
        
        # Send OSC if available
        if self.mixer_client:
            self._send_osc(decisions)
        
        return decisions
    
    def _send_osc(self, decisions: Dict[int, GateDecision]):
        """Send gate decisions via OSC."""
        try:
            for ch, decision in decisions.items():
                # Send gate gain
                if hasattr(self.mixer_client, 'set_channel_gate'):
                    self.mixer_client.set_channel_gate(ch, decision.gain_db)
                    
        except Exception as e:
            logger.error(f"OSC send error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'noise_floor': self.threshold_calculator.noise_floor_db,
            'groups': {
                group.value: len(channels)
                for group, channels in self.group_analyzer.group_channels.items()
            }
        }


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Auto Gate - Cross-Adaptive Intelligent Gate (CAIG) Test")
    print("=" * 70)
    
    # Create controller
    controller = AutoGateController(num_channels=4, sample_rate=48000)
    
    # Configure channels
    controller.configure_channel(0, 'kick', InstrumentGroup.DRUMS, 
                                 attack_ms=0.5, release_ms=80.0, is_drum=True)
    controller.configure_channel(1, 'snare', InstrumentGroup.DRUMS,
                                 attack_ms=0.2, release_ms=50.0, is_drum=True)
    controller.configure_channel(2, 'vocal', InstrumentGroup.VOCALS,
                                 attack_ms=0.5, release_ms=80.0)
    controller.configure_channel(3, 'bass', InstrumentGroup.BASS,
                                 attack_ms=1.0, release_ms=100.0)
    
    # Simulate audio
    print("\nSimulating audio (Kick on ch0, Snare on ch1)...")
    for cycle in range(20):
        audio_data = {}
        
        # Ch0: Kick (loud, periodic)
        t = np.linspace(0, 0.00133, 64)
        if cycle % 4 == 0:  # Kick hits every 4 cycles
            audio_data[0] = 0.8 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 1000)
        else:
            audio_data[0] = np.random.randn(64) * 0.001  # Noise floor
        
        # Ch1: Snare (medium, during kick sometimes)
        if cycle % 4 == 2:  # Snare hits, but suppressed if kick is active
            audio_data[1] = 0.5 * np.random.randn(64)
        else:
            audio_data[1] = np.random.randn(64) * 0.001
        
        # Ch2: Vocal (quiet)
        audio_data[2] = np.random.randn(64) * 0.01
        
        # Ch3: Bass (constant)
        audio_data[3] = 0.3 * np.sin(2 * np.pi * 100 * t)
        
        # Process
        decisions = controller.process(audio_data)
        
        if cycle % 2 == 0:
            print(f"\nCycle {cycle}:")
            for ch, dec in decisions.items():
                inst = controller.instruments.get(ch, '?')
                print(f"  Ch{ch} ({inst:6s}): {dec.state.value:10s} "
                      f"gain={dec.gain_db:5.1f}dB "
                      f"thr={dec.threshold_db:5.1f}dB "
                      f"({'OPEN' if dec.is_triggered else 'closed'})")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
