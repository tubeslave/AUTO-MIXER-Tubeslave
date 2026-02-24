"""
Auto Effects Automation Module - Cross-Adaptive Architecture

Based on Kimi AI debate results:
- 4-Module Architecture: AudioCore, AnalysisEngine, StateManager, OSCInterface
- 13 Audio Features with tiered update frequencies
- Cross-Adaptive Matrix (136 elements for 16 channels)
- OSC Output with deadband and transient bypass

Features (13 total):
Tier 1 (100Hz): RMS, Peak, Crest, OS_Peak, Attack
Tier 2 (50Hz): Loudness, Spectral Centroid  
Tier 3 (30Hz): Spectral Flux, Spectral Spread
Tier 4 (10Hz): Correlation, Flatness, 3-Band Energy
Tier 5 (5Hz): LRA (Loudness Range)
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EffectType(Enum):
    """Types of effects that can be automated."""
    FADER = "fader"
    EQ = "eq"
    COMPRESSOR = "compressor"
    REVERB = "reverb"
    DELAY = "delay"
    PAN = "pan"


class TrackState(Enum):
    """State machine for track processing."""
    SILENT = "silent"      # No signal
    QUIET = "quiet"        # Below threshold
    ACTIVE = "active"      # Normal processing
    LOUD = "loud"          # Approaching limit
    PEAK = "peak"          # Near clipping


@dataclass
class AudioFeatures:
    """Container for all 13 audio features."""
    # Tier 1: 100Hz - Critical amplitude features
    rms: float = 0.0
    peak: float = 0.0
    crest_factor: float = 0.0
    oversampled_peak: float = 0.0
    attack_time_ms: float = 0.0
    
    # Tier 2: 50Hz - Perceptual features
    loudness: float = -70.0  # LUFS
    spectral_centroid: float = 1000.0  # Hz
    
    # Tier 3: 30Hz - Spectral features
    spectral_flux: float = 0.0
    spectral_spread: float = 0.0
    
    # Tier 4: 10Hz - Context features
    correlation: float = 0.0
    spectral_flatness: float = 0.0
    energy_low: float = 0.0
    energy_mid: float = 0.0
    energy_high: float = 0.0
    
    # Tier 5: 5Hz - Slow features
    lra: float = 0.0  # Loudness Range
    
    # Metadata
    timestamp: float = 0.0
    sample_rate: int = 48000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rms': self.rms,
            'peak': self.peak,
            'crest_factor': self.crest_factor,
            'os_peak': self.oversampled_peak,
            'attack_ms': self.attack_time_ms,
            'loudness': self.loudness,
            'centroid': self.spectral_centroid,
            'flux': self.spectral_flux,
            'spread': self.spectral_spread,
            'correlation': self.correlation,
            'flatness': self.spectral_flatness,
            'energy_low': self.energy_low,
            'energy_mid': self.energy_mid,
            'energy_high': self.energy_high,
            'lra': self.lra
        }


class AudioCore:
    """
    Module 1: AudioCore
    
    Responsibilities:
    - Audio input acquisition
    - Feature extraction (13 features)
    - Buffer management
    - Tiered update scheduling
    """
    
    def __init__(self, sample_rate: int = 48000, frame_size: int = 128):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_duration_ms = 1000.0 * frame_size / sample_rate  # ~2.67ms @ 48kHz
        
        # Update counters for tiered processing
        self.update_counters = {
            'tier1': 0,  # Every frame (100Hz)
            'tier2': 0,  # Every 2 frames (50Hz)
            'tier3': 0,  # Every 3 frames (~33Hz)
            'tier4': 0,  # Every 10 frames (10Hz)
            'tier5': 0   # Every 20 frames (5Hz)
        }
        
        # Buffers for feature calculation
        self.rms_buffer = deque(maxlen=10)  # 100ms for RMS smoothing
        self.peak_buffer = deque(maxlen=4)   # 4 frames for peak hold
        self.fft_buffer = deque(maxlen=2048)  # For spectral features
        
        logger.info(f"AudioCore initialized: {sample_rate}Hz, {frame_size} samples/frame")
    
    def process_frame(self, audio: np.ndarray) -> AudioFeatures:
        """
        Process one audio frame and extract features.
        
        Args:
            audio: Audio samples (mono)
            
        Returns:
            AudioFeatures with all 13 features
        """
        features = AudioFeatures()
        features.timestamp = time.time()
        features.sample_rate = self.sample_rate
        
        # Tier 1: Critical features (every frame, 100Hz)
        features = self._extract_tier1_features(audio, features)
        
        # Tier 2: Perceptual features (50Hz)
        if self.update_counters['tier2'] % 2 == 0:
            features = self._extract_tier2_features(audio, features)
        
        # Tier 3: Spectral features (30Hz)
        if self.update_counters['tier3'] % 3 == 0:
            features = self._extract_tier3_features(audio, features)
        
        # Tier 4: Context features (10Hz)
        if self.update_counters['tier4'] % 10 == 0:
            features = self._extract_tier4_features(audio, features)
        
        # Tier 5: Slow features (5Hz)
        if self.update_counters['tier5'] % 20 == 0:
            features = self._extract_tier5_features(audio, features)
        
        # Update counters
        for key in self.update_counters:
            self.update_counters[key] += 1
        
        return features
    
    def _extract_tier1_features(self, audio: np.ndarray, features: AudioFeatures) -> AudioFeatures:
        """Extract critical amplitude features (100Hz)."""
        if audio.size == 0:
            return features
        
        # RMS
        rms_squared = np.mean(audio ** 2)
        rms = np.sqrt(rms_squared + 1e-10)
        self.rms_buffer.append(rms)
        features.rms = 20 * np.log10(np.mean(list(self.rms_buffer)) + 1e-10)
        
        # Peak
        peak = np.max(np.abs(audio))
        self.peak_buffer.append(peak)
        features.peak = 20 * np.log10(np.max(list(self.peak_buffer)) + 1e-10)
        
        # Crest Factor
        if rms > 0:
            features.crest_factor = 20 * np.log10(peak / rms + 1e-10)
        
        # Oversampled Peak (4x)
        upsampled = np.interp(np.linspace(0, len(audio), len(audio) * 4), 
                             np.arange(len(audio)), audio)
        os_peak = np.max(np.abs(upsampled))
        features.oversampled_peak = 20 * np.log10(os_peak + 1e-10)
        
        # Attack Time (simple derivative)
        if len(audio) > 1:
            derivative = np.diff(audio)
            attack_samples = np.argmax(derivative) if np.max(derivative) > 0 else 0
            features.attack_time_ms = 1000.0 * attack_samples / self.sample_rate
        
        return features
    
    def _extract_tier2_features(self, audio: np.ndarray, features: AudioFeatures) -> AudioFeatures:
        """Extract perceptual features (50Hz)."""
        # Add to FFT buffer
        self.fft_buffer.extend(audio)
        
        if len(self.fft_buffer) >= 2048:
            # Get last 2048 samples
            fft_data = np.array(list(self.fft_buffer)[-2048:])
            
            # Window and FFT
            windowed = fft_data * np.hanning(len(fft_data))
            fft_result = np.fft.rfft(windowed)
            magnitude = np.abs(fft_result)
            
            # Spectral Centroid
            freqs = np.fft.rfftfreq(len(fft_data), 1.0 / self.sample_rate)
            if np.sum(magnitude) > 0:
                features.spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            
            # Simple LUFS approximation
            features.loudness = features.rms  # Simplified, use proper LUFS in production
        
        return features
    
    def _extract_tier3_features(self, audio: np.ndarray, features: AudioFeatures) -> AudioFeatures:
        """Extract spectral features (30Hz)."""
        if len(self.fft_buffer) >= 2048:
            # Previous and current FFT for flux
            fft_data = np.array(list(self.fft_buffer)[-2048:])
            windowed = fft_data * np.hanning(len(fft_data))
            magnitude = np.abs(np.fft.rfft(windowed))
            
            # Spectral Flux (change from previous frame)
            if hasattr(self, '_prev_magnitude'):
                diff = magnitude - self._prev_magnitude
                features.spectral_flux = np.sum(np.maximum(0, diff))
            self._prev_magnitude = magnitude
            
            # Spectral Spread
            freqs = np.fft.rfftfreq(len(fft_data), 1.0 / self.sample_rate)
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                variance = np.sum(((freqs - centroid) ** 2) * magnitude) / np.sum(magnitude)
                features.spectral_spread = np.sqrt(variance)
        
        return features
    
    def _extract_tier4_features(self, audio: np.ndarray, features: AudioFeatures) -> AudioFeatures:
        """Extract context features (10Hz)."""
        if len(self.fft_buffer) >= 2048:
            fft_data = np.array(list(self.fft_buffer)[-2048:])
            windowed = fft_data * np.hanning(len(fft_data))
            magnitude = np.abs(np.fft.rfft(windowed))
            
            # Spectral Flatness
            geometric_mean = np.exp(np.mean(np.log(magnitude + 1e-10)))
            arithmetic_mean = np.mean(magnitude)
            if arithmetic_mean > 0:
                features.spectral_flatness = geometric_mean / arithmetic_mean
            
            # 3-Band Energy
            freqs = np.fft.rfftfreq(len(fft_data), 1.0 / self.sample_rate)
            
            # Low: 0-250Hz
            low_mask = freqs < 250
            features.energy_low = np.sum(magnitude[low_mask])
            
            # Mid: 250-4000Hz
            mid_mask = (freqs >= 250) & (freqs < 4000)
            features.energy_mid = np.sum(magnitude[mid_mask])
            
            # High: 4000+ Hz
            high_mask = freqs >= 4000
            features.energy_high = np.sum(magnitude[high_mask])
        
        return features
    
    def _extract_tier5_features(self, audio: np.ndarray, features: AudioFeatures) -> AudioFeatures:
        """Extract slow features (5Hz)."""
        # LRA (Loudness Range) - simplified
        # In production: proper ITU-R BS.1770 LRA calculation
        if len(self.rms_buffer) > 0:
            rms_values = np.array(list(self.rms_buffer))
            features.lra = np.percentile(rms_values, 95) - np.percentile(rms_values, 10)
        
        return features


class AnalysisEngine:
    """
    Module 2: AnalysisEngine
    
    Responsibilities:
    - Cross-adaptive matrix computation (136 elements for 16 channels)
    - Gain Sharing algorithm with adaptive iterations
    - Conflict detection and resolution
    """
    
    def __init__(self, num_channels: int = 16):
        self.num_channels = num_channels
        self.matrix_size = num_channels * (num_channels - 1) // 2  # Upper triangle
        
        # Gain Sharing parameters
        self.alpha = 0.1  # Cross-channel influence
        self.beta = 0.3   # Adaptation speed
        self.hysteresis_db = 0.3
        self.max_iterations = 7
        
        # State
        self.gain_vector = np.ones(num_channels)
        self.coherence_matrix = np.zeros((num_channels, num_channels))
        self.previous_gain = np.ones(num_channels)
        
        logger.info(f"AnalysisEngine initialized: {num_channels} channels, matrix size {self.matrix_size}")
    
    def compute_coherence_matrix(self, features_list: List[AudioFeatures]) -> np.ndarray:
        """
        Compute cross-channel coherence matrix.
        
        Args:
            features_list: List of AudioFeatures for each channel
            
        Returns:
            Coherence matrix (num_channels × num_channels)
        """
        n = len(features_list)
        matrix = np.zeros((n, n))
        
        # Diagonal: channel energy
        for i in range(n):
            matrix[i, i] = 10 ** (features_list[i].rms / 20)
        
        # Upper triangle: cross-correlation
        for i in range(n):
            for j in range(i + 1, n):
                # Simplified correlation based on RMS similarity
                rms_i = 10 ** (features_list[i].rms / 20)
                rms_j = 10 ** (features_list[j].rms / 20)
                
                if rms_i > 0 and rms_j > 0:
                    # Correlation based on spectral similarity
                    centroid_diff = abs(features_list[i].spectral_centroid - 
                                      features_list[j].spectral_centroid)
                    correlation = np.exp(-centroid_diff / 2000)  # Decay with frequency diff
                    
                    matrix[i, j] = correlation * np.sqrt(rms_i * rms_j)
                    matrix[j, i] = matrix[i, j]  # Symmetric
        
        self.coherence_matrix = matrix
        return matrix
    
    def gain_sharing(self, desired_levels_db: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Iterative Gain Sharing algorithm.
        
        Args:
            desired_levels_db: Target level for each channel in dB
            
        Returns:
            (gain_vector, iterations_used)
        """
        desired_linear = 10 ** (desired_levels_db / 20)
        gain = self.gain_vector.copy()
        
        for iteration in range(self.max_iterations):
            previous_gain = gain.copy()
            
            for i in range(self.num_channels):
                # Calculate interference from other channels
                interference = 0.0
                for j in range(self.num_channels):
                    if i != j:
                        interference += (self.coherence_matrix[i, j] * gain[j])
                
                # Target gain considering interference
                target_gain = desired_linear[i] / (self.coherence_matrix[i, i] + 
                                                   self.alpha * interference + 1e-10)
                
                # Smooth adaptation
                gain[i] = (1 - self.beta) * gain[i] + self.beta * target_gain
            
            # Check convergence
            delta = np.max(np.abs(gain - previous_gain))
            if delta < 0.01:  # 0.1 dB
                self.gain_vector = gain
                return gain, iteration + 1
            
            # Check divergence
            if delta > np.max(np.abs(previous_gain - self.previous_gain)) * 1.5:
                logger.warning(f"Gain Sharing diverging at iteration {iteration}, using previous")
                return self.previous_gain, iteration + 1
        
        self.previous_gain = gain.copy()
        self.gain_vector = gain
        return gain, self.max_iterations
    
    def detect_conflicts(self, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Detect frequency conflicts between channels.
        
        Returns:
            List of (channel_i, channel_j, severity) tuples
        """
        conflicts = []
        
        for i in range(self.num_channels):
            for j in range(i + 1, self.num_channels):
                if self.coherence_matrix[i, j] > threshold:
                    conflicts.append((i, j, self.coherence_matrix[i, j]))
        
        return conflicts


class StateManager:
    """
    Module 3: StateManager
    
    Responsibilities:
    - Track state machine (6 states)
    - Parameter mapping
    - History tracking
    """
    
    def __init__(self, num_channels: int = 16):
        self.num_channels = num_channels
        self.track_states: Dict[int, TrackState] = {}
        self.parameter_history: Dict[int, deque] = {}
        
        # Initialize
        for i in range(num_channels):
            self.track_states[i] = TrackState.SILENT
            self.parameter_history[i] = deque(maxlen=100)
        
        logger.info(f"StateManager initialized for {num_channels} channels")
    
    def update_state(self, channel_id: int, features: AudioFeatures) -> TrackState:
        """Update track state based on features."""
        # Thresholds in dB
        if features.rms < -70:
            new_state = TrackState.SILENT
        elif features.rms < -50:
            new_state = TrackState.QUIET
        elif features.rms < -20:
            new_state = TrackState.ACTIVE
        elif features.rms < -10:
            new_state = TrackState.LOUD
        else:
            new_state = TrackState.PEAK
        
        # Hysteresis to prevent state fluttering
        old_state = self.track_states[channel_id]
        if abs(features.rms - self._state_threshold(old_state)) < 3.0:
            new_state = old_state
        
        self.track_states[channel_id] = new_state
        return new_state
    
    def _state_threshold(self, state: TrackState) -> float:
        """Get threshold for state."""
        thresholds = {
            TrackState.SILENT: -70,
            TrackState.QUIET: -50,
            TrackState.ACTIVE: -20,
            TrackState.LOUD: -10,
            TrackState.PEAK: 0
        }
        return thresholds.get(state, -50)
    
    def map_to_parameters(self, channel_id: int, features: AudioFeatures, 
                         gain: float) -> Dict[str, float]:
        """Map features to effect parameters."""
        params = {}
        
        # Fader: based on gain from cross-adaptive
        params['fader_db'] = 20 * np.log10(gain + 1e-10)
        
        # EQ: based on spectral features
        params['eq_low_gain'] = self._db_to_linear(features.energy_low)
        params['eq_mid_gain'] = self._db_to_linear(features.energy_mid)
        params['eq_high_gain'] = self._db_to_linear(features.energy_high)
        
        # Compressor: based on crest factor and LRA
        params['comp_threshold'] = -20 - features.crest_factor
        params['comp_ratio'] = min(10.0, 1.0 + features.lra / 5.0)
        
        # Reverb: based on spectral spread and correlation
        params['reverb_amount'] = 1.0 - features.correlation
        params['reverb_decay'] = 0.5 + features.spectral_spread / 10000
        
        self.parameter_history[channel_id].append(params)
        return params
    
    def _db_to_linear(self, db: float) -> float:
        """Convert dB to linear."""
        return 10 ** (db / 20)


class OSCInterface:
    """
    Module 4: OSCInterface
    
    Responsibilities:
    - OSC message formatting
    - Rate limiting with deadband
    - Transient bypass
    - Packet transmission
    """
    
    def __init__(self, deadband: float = 0.02, update_freq_hz: float = 60.0):
        self.deadband = deadband
        self.update_period = 1.0 / update_freq_hz
        
        # Last sent values for deadband comparison
        self.last_sent: Dict[str, float] = {}
        self.last_update_time: Dict[str, float] = {}
        
        # Transient detection
        self.transient_threshold = 0.1  # 10% change
        
        logger.info(f"OSCInterface initialized: deadband={deadband}, {update_freq_hz}Hz")
    
    def should_send(self, address: str, value: float, is_transient: bool = False) -> bool:
        """
        Check if value should be sent (deadband logic).
        
        Args:
            address: OSC address
            value: Current value
            is_transient: If True, bypass deadband
            
        Returns:
            True if should send
        """
        current_time = time.time()
        
        # Transient bypass
        if is_transient:
            self.last_sent[address] = value
            self.last_update_time[address] = current_time
            return True
        
        # Check time
        if address in self.last_update_time:
            if current_time - self.last_update_time[address] < self.update_period:
                return False
        
        # Check deadband
        if address in self.last_sent:
            relative_change = abs(value - self.last_sent[address]) / (abs(self.last_sent[address]) + 1e-10)
            if relative_change < self.deadband:
                return False
        
        self.last_sent[address] = value
        self.last_update_time[address] = current_time
        return True
    
    def format_message(self, track_id: int, feature_name: str, value: float) -> Tuple[str, float]:
        """Format OSC address and value."""
        address = f"/track/{track_id}/{feature_name}"
        return address, value
    
    def format_cross_matrix(self, matrix: np.ndarray) -> Tuple[str, List[float]]:
        """Format cross-adaptive matrix for OSC."""
        # Send as row-wise list
        values = []
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                values.append(matrix[i, j])
        
        return "/cross/matrix", values


class AutoEffectsController:
    """
    Main Controller for Auto Effects Automation.
    
    Integrates all 4 modules:
    1. AudioCore: Feature extraction
    2. AnalysisEngine: Cross-adaptive processing
    3. StateManager: State machine and mapping
    4. OSCInterface: Output with rate limiting
    """
    
    def __init__(self, num_channels: int = 16, sample_rate: int = 48000,
                 mixer_client=None):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.mixer_client = mixer_client
        
        # Initialize modules
        self.audio_core = AudioCore(sample_rate)
        self.analysis_engine = AnalysisEngine(num_channels)
        self.state_manager = StateManager(num_channels)
        self.osc_interface = OSCInterface()
        
        # Track data
        self.channel_features: Dict[int, AudioFeatures] = {}
        
        logger.info(f"AutoEffectsController initialized: {num_channels} channels")
    
    def process(self, audio_data: Dict[int, np.ndarray]) -> Dict[int, Dict[str, Any]]:
        """
        Process one cycle of audio data.
        
        Args:
            audio_data: {channel_id: audio_samples}
            
        Returns:
            Processing results for each channel
        """
        results = {}
        features_list = []
        
        # Step 1: AudioCore - extract features
        for channel_id, audio in audio_data.items():
            if channel_id < self.num_channels:
                features = self.audio_core.process_frame(audio)
                self.channel_features[channel_id] = features
                features_list.append(features)
        
        # Pad features list if needed
        while len(features_list) < self.num_channels:
            features_list.append(AudioFeatures())
        
        # Step 2: AnalysisEngine - cross-adaptive processing
        self.analysis_engine.compute_coherence_matrix(features_list)
        
        desired_levels = np.array([f.rms for f in features_list])
        gains, iterations = self.analysis_engine.gain_sharing(desired_levels)
        
        conflicts = self.analysis_engine.detect_conflicts()
        
        # Step 3 & 4: StateManager + OSCInterface
        for channel_id in range(self.num_channels):
            features = self.channel_features.get(channel_id, AudioFeatures())
            
            # Update state
            state = self.state_manager.update_state(channel_id, features)
            
            # Map to parameters
            params = self.state_manager.map_to_parameters(channel_id, features, 
                                                         gains[channel_id])
            
            results[channel_id] = {
                'features': features.to_dict(),
                'state': state.value,
                'gain': gains[channel_id],
                'parameters': params,
                'iterations': iterations if channel_id == 0 else None
            }
            
            # Send OSC if in full-auto mode
            if self.mixer_client:
                self._send_osc(channel_id, features, params)
        
        return results
    
    def _send_osc(self, channel_id: int, features: AudioFeatures, params: Dict[str, float]):
        """Send OSC commands for channel."""
        try:
            # Features (with deadband)
            feature_map = {
                'rms': features.rms,
                'peak': features.peak,
                'loudness': features.loudness,
                'centroid': features.spectral_centroid,
                'flux': features.spectral_flux,
                'spread': features.spectral_spread
            }
            
            for name, value in feature_map.items():
                address, val = self.osc_interface.format_message(channel_id, name, value)
                if self.osc_interface.should_send(address, val):
                    # In production: self.mixer_client.send_osc(address, val)
                    pass
            
            # Parameters (with deadband, transient bypass for fader)
            param_map = {
                'fader': params['fader_db'],
                'eq_low': params['eq_low_gain'],
                'eq_mid': params['eq_mid_gain'],
                'eq_high': params['eq_high_gain']
            }
            
            for name, value in param_map.items():
                address, val = self.osc_interface.format_message(channel_id, 
                                                                  f"effect/{name}", value)
                is_transient = (name == 'fader' and abs(value) > 3.0)
                if self.osc_interface.should_send(address, val, is_transient):
                    # In production: self.mixer_client.send_osc(address, val)
                    pass
                    
        except Exception as e:
            logger.error(f"OSC send error for ch{channel_id}: {e}")


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Auto Effects Automation - Test")
    print("=" * 70)
    
    # Create controller
    controller = AutoEffectsController(num_channels=4, sample_rate=48000)
    
    # Simulate audio input
    print("\nSimulating 4 channels of audio...")
    for cycle in range(10):
        audio_data = {}
        for ch in range(4):
            # Generate test signal
            t = np.linspace(0, 0.01, 128)  # 10ms @ 48kHz
            freq = 440 * (ch + 1)
            signal = 0.5 * np.sin(2 * np.pi * freq * t) * (0.5 + 0.5 * np.random.random())
            audio_data[ch] = signal
        
        # Process
        results = controller.process(audio_data)
        
        if cycle % 3 == 0:
            print(f"\nCycle {cycle}:")
            for ch, data in results.items():
                print(f"  Ch{ch}: RMS={data['features']['rms']:5.1f}dB, "
                      f"Gain={data['gain']:.2f}, State={data['state']}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
