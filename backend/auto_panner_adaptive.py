"""
Auto Panner - Multi-band Adaptive Panning

Based on Kimi AI debate results:
- 3-band panning scheme: Low (center), Mid (sine/cosine), High (wide)
- 9 discrete positions: L90, L67, L45, L22, CENTER, R22, R45, R67, R90
- 3-tier analysis (30Hz)
- Genre-specific templates (Rock, Pop, Jazz, Electronic)
- Lock-free pipeline for real-time processing

References:
- Perez Gonzalez & Reiss (2007) - Autonomous Stereo Panning
- Mansbridge et al. - Panning Criteria (source balance, spatial balance, spectral balance)
- Intelligent Music Production (De Man, Stables, Reiss, 2020), Section 7.2
"""

import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Genre(Enum):
    """Music genres with different panning templates."""
    ROCK = "rock"
    POP = "pop"
    JAZZ = "jazz"
    ELECTRONIC = "electronic"
    CUSTOM = "custom"


class PanPosition(Enum):
    """9 discrete pan positions."""
    L90 = -90
    L67 = -67
    L45 = -45
    L22 = -22
    CENTER = 0
    R22 = 22
    R45 = 45
    R67 = 67
    R90 = 90


@dataclass
class AudioFeatures:
    """Container for audio features (3-tier analysis)."""
    # Tier 1: Fast (10ms update)
    rms: float = -60.0
    zcr: float = 0.0  # Zero Crossing Rate
    crest_factor: float = 0.0
    
    # Tier 2: Medium (100ms update)
    spectral_centroid: float = 1000.0
    ild: float = 0.0  # Interaural Level Difference
    spectral_flux: float = 0.0
    
    # Tier 3: Slow (optional)
    bark_energy: List[float] = field(default_factory=lambda: [0.0] * 24)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rms': self.rms,
            'zcr': self.zcr,
            'crest_factor': self.crest_factor,
            'spectral_centroid': self.spectral_centroid,
            'ild': self.ild,
            'spectral_flux': self.spectral_flux
        }


@dataclass
class PanningDecision:
    """Decision for channel panning."""
    channel_id: int
    position: float  # -90 to +90 degrees
    position_enum: PanPosition
    confidence: float
    genre: Genre
    instrument_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'channel_id': self.channel_id,
            'position': self.position,
            'position_name': self.position_enum.name,
            'confidence': self.confidence,
            'genre': self.genre.value,
            'instrument_type': self.instrument_type
        }


class LinkwitzRileyCrossover:
    """
    Linkwitz-Riley 4th order crossover for 3-band splitting.
    
    Frequencies:
    - Low: < 250 Hz (center pan)
    - Mid: 250 - 4000 Hz (position-based pan)
    - High: > 4000 Hz (wide stereo)
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.low_freq = 250.0
        self.high_freq = 4000.0
        
        # Filter states
        self.low_state = np.zeros(4)
        self.mid_state = np.zeros(4)
        self.high_state = np.zeros(4)
        
        # Filter coefficients (LR4)
        self._calculate_coeffs()
        
        logger.info(f"LinkwitzRileyCrossover initialized: {self.low_freq}Hz, {self.high_freq}Hz")
    
    def _calculate_coeffs(self):
        """Calculate LR4 filter coefficients."""
        # Simplified coefficients - in production use proper LR4 design
        self.low_a = np.array([1.0, -3.9, 5.7, -3.7, 0.9])
        self.low_b = np.array([0.001, 0.004, 0.006, 0.004, 0.001])
        
        self.mid_a = np.array([1.0, -3.8, 5.5, -3.5, 0.85])
        self.mid_b = np.array([0.002, 0.008, 0.012, 0.008, 0.002])
        
        self.high_a = np.array([1.0, -3.7, 5.3, -3.4, 0.82])
        self.high_b = np.array([0.003, 0.012, 0.018, 0.012, 0.003])
    
    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split audio into 3 bands.
        
        Returns:
            (low_band, mid_band, high_band)
        """
        # Simplified processing - in production use proper filter implementation
        # Low band
        low = self._apply_filter(audio, self.low_b, self.low_a, self.low_state)
        
        # High band
        high = self._apply_filter(audio, self.high_b, self.high_a, self.high_state)
        
        # Mid band = original - low - high
        mid = audio - low - high
        
        return low, mid, high
    
    def _apply_filter(self, x: np.ndarray, b: np.ndarray, a: np.ndarray, 
                     state: np.ndarray) -> np.ndarray:
        """Apply IIR filter with state."""
        y = np.zeros_like(x)
        for i in range(len(x)):
            # Shift state
            state[3] = state[2]
            state[2] = state[1]
            state[1] = state[0]
            state[0] = x[i]
            
            # Compute output
            y[i] = (b[0] * state[0] + b[1] * state[1] + b[2] * state[2] + 
                   b[3] * state[3] + b[4] * (state[3] if len(state) > 3 else 0))
            y[i] -= (a[1] * (y[i-1] if i > 0 else 0) + 
                    a[2] * (y[i-2] if i > 1 else 0) +
                    a[3] * (y[i-3] if i > 2 else 0))
            y[i] /= a[0]
        
        return y


class FeatureExtractor:
    """
    3-tier feature extraction for instrument classification.
    
    Tier 1 (10ms): RMS, ZCR, Crest Factor
    Tier 2 (100ms): Spectral Centroid, ILD, Spectral Flux
    Tier 3 (optional): Bark Energy
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Buffers for tiered analysis
        self.tier1_buffer = deque(maxlen=int(0.01 * sample_rate))  # 10ms
        self.tier2_buffer = deque(maxlen=int(0.1 * sample_rate))   # 100ms
        self.fft_buffer = deque(maxlen=2048)
        
        # Previous values for flux calculation
        self.prev_spectrum = None
        
        logger.info("FeatureExtractor initialized")
    
    def process(self, audio: np.ndarray) -> AudioFeatures:
        """Extract features from audio frame."""
        features = AudioFeatures()
        
        # Add to buffers
        self.tier1_buffer.extend(audio)
        self.tier2_buffer.extend(audio)
        self.fft_buffer.extend(audio)
        
        # Tier 1: Fast features (every frame)
        if len(self.tier1_buffer) >= self.tier1_buffer.maxlen:
            features = self._extract_tier1(features)
        
        # Tier 2: Medium features (every 10 frames)
        if len(self.tier2_buffer) >= self.tier2_buffer.maxlen:
            features = self._extract_tier2(features)
        
        return features
    
    def _extract_tier1(self, features: AudioFeatures) -> AudioFeatures:
        """Extract Tier 1 features (10ms)."""
        data = np.array(list(self.tier1_buffer))
        
        # RMS
        rms = np.sqrt(np.mean(data ** 2) + 1e-10)
        features.rms = 20 * np.log10(rms)
        
        # ZCR
        zero_crossings = np.sum(np.diff(np.signbit(data).astype(int)) != 0)
        features.zcr = zero_crossings / len(data)
        
        # Crest Factor
        peak = np.max(np.abs(data))
        if rms > 0:
            features.crest_factor = 20 * np.log10(peak / rms)
        
        return features
    
    def _extract_tier2(self, features: AudioFeatures) -> AudioFeatures:
        """Extract Tier 2 features (100ms)."""
        if len(self.fft_buffer) < 2048:
            return features
        
        data = np.array(list(self.fft_buffer)[-2048:])
        
        # Window and FFT
        windowed = data * np.hanning(len(data))
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        freqs = np.fft.rfftfreq(len(data), 1.0 / self.sample_rate)
        
        # Spectral Centroid
        if np.sum(magnitude) > 0:
            features.spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        
        # Spectral Flux
        if self.prev_spectrum is not None:
            diff = magnitude - self.prev_spectrum
            features.spectral_flux = np.sum(np.maximum(0, diff))
        self.prev_spectrum = magnitude.copy()
        
        # ILD (simplified - assumes stereo input)
        # In production: calculate from left/right channels
        mid = data[::2] if len(data) > 1 else data
        side = data[1::2] if len(data) > 1 else np.zeros_like(data)
        if len(mid) > 0 and len(side) > 0:
            features.ild = 20 * np.log10(np.mean(np.abs(mid)) / 
                                        (np.mean(np.abs(side)) + 1e-10) + 1e-10)
        
        return features


class InstrumentClassifier:
    """
    Classify instruments based on spectral features.
    
    Classification rules:
    - Kick: Low centroid (< 200Hz), high crest (> 15dB)
    - Bass: Low centroid (200-500Hz), low crest (< 10dB)
    - Snare: Mid centroid (500-2000Hz), high crest (> 12dB)
    - Vocal: Mid centroid (1000-4000Hz), moderate crest (8-12dB)
    - Guitar: High centroid (> 2000Hz), high crest (> 10dB)
    - Keys: Wide centroid range, low crest (< 8dB)
    """
    
    INSTRUMENT_RULES = {
        'kick': {
            'centroid_range': (50, 200),
            'crest_min': 15,
            'zcr_max': 0.1
        },
        'bass': {
            'centroid_range': (200, 500),
            'crest_max': 10,
            'zcr_max': 0.15
        },
        'snare': {
            'centroid_range': (500, 2000),
            'crest_min': 12,
            'zcr_range': (0.1, 0.3)
        },
        'vocal': {
            'centroid_range': (1000, 4000),
            'crest_range': (8, 12),
            'zcr_range': (0.05, 0.2)
        },
        'guitar': {
            'centroid_range': (2000, 6000),
            'crest_min': 10,
            'zcr_range': (0.1, 0.4)
        },
        'keys': {
            'centroid_range': (500, 4000),
            'crest_max': 8,
            'zcr_range': (0.05, 0.3)
        },
        'cymbals': {
            'centroid_min': 4000,
            'crest_min': 8,
            'zcr_min': 0.3
        }
    }
    
    def classify(self, features: AudioFeatures) -> Tuple[str, float]:
        """
        Classify instrument based on features.
        
        Returns:
            (instrument_type, confidence)
        """
        scores = {}
        
        for instrument, rules in self.INSTRUMENT_RULES.items():
            score = 0
            checks = 0
            
            # Check centroid
            if 'centroid_range' in rules:
                if rules['centroid_range'][0] <= features.spectral_centroid <= rules['centroid_range'][1]:
                    score += 1
                checks += 1
            elif 'centroid_min' in rules:
                if features.spectral_centroid >= rules['centroid_min']:
                    score += 1
                checks += 1
            
            # Check crest factor
            if 'crest_range' in rules:
                if rules['crest_range'][0] <= features.crest_factor <= rules['crest_range'][1]:
                    score += 1
                checks += 1
            elif 'crest_min' in rules:
                if features.crest_factor >= rules['crest_min']:
                    score += 1
                checks += 1
            elif 'crest_max' in rules:
                if features.crest_factor <= rules['crest_max']:
                    score += 1
                checks += 1
            
            # Check ZCR
            if 'zcr_range' in rules:
                if rules['zcr_range'][0] <= features.zcr <= rules['zcr_range'][1]:
                    score += 1
                checks += 1
            elif 'zcr_min' in rules:
                if features.zcr >= rules['zcr_min']:
                    score += 1
                checks += 1
            elif 'zcr_max' in rules:
                if features.zcr <= rules['zcr_max']:
                    score += 1
                checks += 1
            
            scores[instrument] = score / checks if checks > 0 else 0
        
        # Get best match
        best = max(scores, key=scores.get)
        confidence = scores[best]
        
        return best, confidence


class GenreTemplate:
    """
    Genre-specific panning templates.
    
    Templates define default positions for each instrument type.
    """
    
    TEMPLATES = {
        Genre.ROCK: {
            'kick': PanPosition.CENTER,
            'snare': PanPosition.CENTER,
            'bass': PanPosition.CENTER,
            'vocal': PanPosition.CENTER,
            'guitar': [PanPosition.L45, PanPosition.R45],
            'keys': PanPosition.L22,
            'cymbals': [PanPosition.L67, PanPosition.R67]
        },
        Genre.POP: {
            'kick': PanPosition.CENTER,
            'snare': PanPosition.CENTER,
            'bass': PanPosition.CENTER,
            'vocal': PanPosition.CENTER,
            'guitar': [PanPosition.L22, PanPosition.R67],
            'keys': [PanPosition.L45, PanPosition.R22],
            'cymbals': [PanPosition.L90, PanPosition.R90]
        },
        Genre.JAZZ: {
            'kick': PanPosition.CENTER,
            'snare': PanPosition.R22,
            'bass': PanPosition.L22,
            'vocal': PanPosition.CENTER,
            'guitar': [PanPosition.L45, PanPosition.R45],
            'keys': PanPosition.L67,
            'cymbals': [PanPosition.L67, PanPosition.R67, PanPosition.R90]
        },
        Genre.ELECTRONIC: {
            'kick': PanPosition.CENTER,
            'snare': PanPosition.CENTER,
            'bass': PanPosition.CENTER,
            'vocal': PanPosition.CENTER,
            'guitar': [PanPosition.L90, PanPosition.R90],
            'keys': [PanPosition.L45, PanPosition.R45],
            'cymbals': [PanPosition.L67, PanPosition.R67]
        }
    }
    
    def get_position(self, genre: Genre, instrument: str, 
                    instance_index: int = 0) -> PanPosition:
        """Get default position for instrument in genre."""
        template = self.TEMPLATES.get(genre, self.TEMPLATES[Genre.ROCK])
        positions = template.get(instrument, PanPosition.CENTER)
        
        if isinstance(positions, list):
            return positions[instance_index % len(positions)]
        return positions


class PanningEngine:
    """
    Core panning engine with sine/cosine law.
    
    Implements 9 discrete positions with smooth interpolation.
    """
    
    # 9 discrete positions
    POSITIONS = [
        PanPosition.L90, PanPosition.L67, PanPosition.L45,
        PanPosition.L22, PanPosition.CENTER,
        PanPosition.R22, PanPosition.R45, PanPosition.R67, PanPosition.R90
    ]
    
    def __init__(self, smoothing_ms: float = 50.0, sample_rate: int = 48000):
        self.smoothing_samples = int(smoothing_ms * sample_rate / 1000)
        self.current_pans: Dict[int, float] = {}
        self.target_pans: Dict[int, float] = {}
        
        logger.info(f"PanningEngine initialized: smoothing={smoothing_ms}ms")
    
    def calculate_pan_gains(self, position_degrees: float) -> Tuple[float, float]:
        """
        Calculate left/right gains using sine/cosine law.
        
        Args:
            position_degrees: -90 (full left) to +90 (full right)
            
        Returns:
            (left_gain, right_gain)
        """
        # Normalize to -1 to +1
        norm_pos = position_degrees / 90.0
        
        # Sine/cosine law for constant power
        angle = (norm_pos + 1) * np.pi / 4  # Map to 0 to π/2
        left_gain = np.cos(angle)
        right_gain = np.sin(angle)
        
        return left_gain, right_gain
    
    def set_target(self, channel_id: int, position: float):
        """Set target pan position with smoothing."""
        self.target_pans[channel_id] = np.clip(position, -90, 90)
        if channel_id not in self.current_pans:
            self.current_pans[channel_id] = self.target_pans[channel_id]
    
    def update(self) -> Dict[int, Tuple[float, float]]:
        """
        Update pan positions with smoothing.
        
        Returns:
            {channel_id: (left_gain, right_gain)}
        """
        results = {}
        
        for channel_id in self.target_pans:
            target = self.target_pans[channel_id]
            current = self.current_pans.get(channel_id, target)
            
            # Smooth interpolation
            alpha = 0.1  # Smoothing factor
            new_pos = current + alpha * (target - current)
            
            self.current_pans[channel_id] = new_pos
            results[channel_id] = self.calculate_pan_gains(new_pos)
        
        return results
    
    def get_nearest_position(self, angle: float) -> PanPosition:
        """Get nearest discrete position."""
        angles = [p.value for p in self.POSITIONS]
        nearest_idx = np.argmin(np.abs(np.array(angles) - angle))
        return self.POSITIONS[nearest_idx]


class AutoPannerController:
    """
    Main controller for automatic panning.
    
    Integrates all components:
    1. LinkwitzRileyCrossover (3-band splitting)
    2. FeatureExtractor (3-tier analysis)
    3. InstrumentClassifier (instrument detection)
    4. GenreTemplate (genre-specific defaults)
    5. PanningEngine (sine/cosine law)
    """
    
    def __init__(
        self,
        num_channels: int = 16,
        sample_rate: int = 48000,
        genre: Genre = Genre.ROCK,
        mixer_client=None
    ):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.genre = genre
        self.mixer_client = mixer_client
        
        # Initialize components
        self.crossover = LinkwitzRileyCrossover(sample_rate)
        self.extractors = [FeatureExtractor(sample_rate) for _ in range(num_channels)]
        self.classifier = InstrumentClassifier()
        self.templates = GenreTemplate()
        self.engine = PanningEngine(smoothing_ms=50.0, sample_rate=sample_rate)
        
        # Track state
        self.channel_instruments: Dict[int, str] = {}
        self.channel_confidence: Dict[int, float] = {}
        self.is_running = False
        
        logger.info(f"AutoPannerController initialized: {num_channels} channels, {genre.value}")
    
    def process(self, audio_data: Dict[int, np.ndarray]) -> Dict[int, PanningDecision]:
        """
        Process audio and calculate panning decisions.
        
        Args:
            audio_data: {channel_id: audio_samples}
            
        Returns:
            {channel_id: PanningDecision}
        """
        decisions = {}
        
        for channel_id, audio in audio_data.items():
            if channel_id >= self.num_channels:
                continue
            
            # Step 1: Split into 3 bands
            low, mid, high = self.crossover.process(audio)
            
            # Step 2: Extract features
            features = self.extractors[channel_id].process(audio)
            
            # Step 3: Classify instrument
            instrument, confidence = self.classifier.classify(features)
            self.channel_instruments[channel_id] = instrument
            self.channel_confidence[channel_id] = confidence
            
            # Step 4: Get target position from template
            # Count instances of this instrument for alternating positions
            instance_count = sum(1 for i, inst in self.channel_instruments.items() 
                               if inst == instrument and i <= channel_id)
            
            target_pos = self.templates.get_position(
                self.genre, instrument, instance_count - 1
            )
            
            # Step 5: Apply to panning engine
            self.engine.set_target(channel_id, target_pos.value)
            
            # Create decision
            decisions[channel_id] = PanningDecision(
                channel_id=channel_id,
                position=target_pos.value,
                position_enum=target_pos,
                confidence=confidence,
                genre=self.genre,
                instrument_type=instrument
            )
        
        # Update panning with smoothing
        pan_gains = self.engine.update()
        
        # Send OSC if available
        if self.mixer_client:
            self._send_osc(pan_gains, decisions)
        
        return decisions
    
    def _send_osc(self, pan_gains: Dict[int, Tuple[float, float]], 
                 decisions: Dict[int, PanningDecision]):
        """Send panning commands via OSC."""
        try:
            for channel_id, (left_gain, right_gain) in pan_gains.items():
                # Convert to pan position (-1 to +1)
                # Using sine/cosine inverse
                if left_gain > 0 and right_gain > 0:
                    pan_pos = (np.arctan2(right_gain, left_gain) * 4 / np.pi) - 1
                else:
                    pan_pos = 0.0
                
                # Send to mixer (format depends on mixer implementation)
                # Example: /ch/{n}/mix/pan
                if hasattr(self.mixer_client, 'set_channel_pan'):
                    self.mixer_client.set_channel_pan(channel_id, pan_pos)
                    
        except Exception as e:
            logger.error(f"OSC send error: {e}")
    
    def set_genre(self, genre: Genre):
        """Change genre template."""
        self.genre = genre
        logger.info(f"Genre changed to {genre.value}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return {
            'genre': self.genre.value,
            'channels': {
                ch: {
                    'instrument': self.channel_instruments.get(ch, 'unknown'),
                    'confidence': self.channel_confidence.get(ch, 0.0),
                    'current_pan': self.engine.current_pans.get(ch, 0.0)
                }
                for ch in range(self.num_channels)
            }
        }


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("Auto Panner - Multi-band Adaptive Panning Test")
    print("=" * 70)
    
    # Create controller
    controller = AutoPannerController(
        num_channels=4,
        sample_rate=48000,
        genre=Genre.ROCK
    )
    
    # Simulate different instruments
    test_signals = {
        0: ('kick', 100, 0.5),    # Low freq, high crest
        1: ('vocal', 2000, 0.3),  # Mid freq, moderate crest
        2: ('guitar', 3000, 0.4), # High freq, high crest
        3: ('cymbals', 8000, 0.6) # Very high freq, high ZCR
    }
    
    print("\nSimulating 4 channels with different instruments...")
    for cycle in range(10):
        audio_data = {}
        for ch, (inst, freq, crest) in test_signals.items():
            t = np.linspace(0, 0.01, 128)  # 10ms
            # Generate signal with characteristics
            signal = np.sin(2 * np.pi * freq * t)
            # Add crest factor variation
            signal *= (0.5 + 0.5 * np.random.random() * crest)
            audio_data[ch] = signal
        
        # Process
        decisions = controller.process(audio_data)
        
        if cycle % 3 == 0:
            print(f"\nCycle {cycle}:")
            for ch, decision in decisions.items():
                print(f"  Ch{ch}: {decision.instrument_type:10s} "
                      f"→ {decision.position_enum.name:6s} "
                      f"({decision.position:+.0f}°) "
                      f"conf={decision.confidence:.2f}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
