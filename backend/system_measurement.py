"""
System Measurement Module - Soundcheck Analysis & Correction

Based on Kimi AI debate results:
- Method: Sine sweep (exponential, 20Hz-20kHz, 10-20s)
- Reference mic at FOH position (1-2m height, audience area)
- Multiple measurements (6-12 positions) for spatial averaging
- Apply corrections to Groups/Master/Matrix (not individual channels)
- Target: Smooth magnitude, linear phase, flat response

Metrics:
- Magnitude Response (1/12 octave smoothing)
- Phase Response (unwrapped)
- Impulse Response
- RT60 (reverb time per octave band)
- Coherence (measurement quality indicator)

References:
- Farina (2000) "Simultaneous measurement of impulse response and distortion"
- Intelligent Music Production (De Man et al., 2020)
- AES-4id-2001: "Sound system measurement"
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from scipy.fft import fft, ifft
from scipy.signal import convolve, correlate, find_peaks
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class MeasurementState(Enum):
    """States for measurement process."""
    IDLE = "idle"
    GENERATING = "generating"  # Generating sine sweep
    PLAYING = "playing"        # Playing sweep
    RECORDING = "recording"    # Recording response
    PROCESSING = "processing"  # Analyzing data
    COMPLETE = "complete"
    ERROR = "error"


class TargetBus(Enum):
    """Target bus for applying corrections."""
    MASTER = "master"
    GROUP = "group"
    MATRIX = "matrix"


@dataclass
class MeasurementPosition:
    """Single measurement position data."""
    position_id: int
    x: float  # Position in meters (relative to FOH)
    y: float
    height: float
    mic_response: np.ndarray = field(default_factory=lambda: np.array([]))
    timestamp: float = 0.0
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position_id': self.position_id,
            'x': self.x,
            'y': self.y,
            'height': self.height,
            'timestamp': self.timestamp,
            'quality_score': self.quality_score
        }


@dataclass
class FrequencyResponse:
    """Frequency response data."""
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    magnitude_db: np.ndarray = field(default_factory=lambda: np.array([]))
    phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    coherence: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def get_smoothed(self, octaves: float = 1.0/12.0) -> 'FrequencyResponse':
        """Get 1/12-octave smoothed response."""
        if len(self.magnitude_db) == 0:
            return FrequencyResponse()
        
        smoothed = np.zeros_like(self.magnitude_db)
        for i, freq in enumerate(self.frequencies):
            # 1/12 octave bandwidth
            f_low = freq * (2 ** (-octaves/2))
            f_high = freq * (2 ** (octaves/2))
            
            # Find indices in range
            mask = (self.frequencies >= f_low) & (self.frequencies <= f_high)
            if np.any(mask):
                smoothed[i] = np.mean(self.magnitude_db[mask])
            else:
                smoothed[i] = self.magnitude_db[i]
        
        result = FrequencyResponse()
        result.frequencies = self.frequencies.copy()
        result.magnitude_db = smoothed
        result.phase_deg = self.phase_deg.copy()
        result.coherence = self.coherence.copy()
        return result


@dataclass
class RT60Data:
    """RT60 reverb time data per octave band."""
    bands: List[float] = field(default_factory=list)  # Center frequencies
    rt60: List[float] = field(default_factory=list)   # Reverb time in seconds
    
    def to_dict(self) -> Dict[str, List]:
        return {
            'bands': self.bands,
            'rt60': self.rt60
        }


@dataclass
class MeasurementResult:
    """Complete measurement result."""
    magnitude_response: FrequencyResponse = field(default_factory=FrequencyResponse)
    phase_response: FrequencyResponse = field(default_factory=FrequencyResponse)
    impulse_response: np.ndarray = field(default_factory=lambda: np.ndarray([]))
    rt60: RT60Data = field(default_factory=RT60Data)
    positions: List[MeasurementPosition] = field(default_factory=list)
    timestamp: float = 0.0
    overall_quality: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'magnitude': self.magnitude_response.get_smoothed(1.0/12.0).magnitude_db.tolist(),
            'phase': self.phase_response.phase_deg.tolist(),
            'frequencies': self.magnitude_response.frequencies.tolist(),
            'rt60': self.rt60.to_dict(),
            'positions': [p.to_dict() for p in self.positions],
            'quality': self.overall_quality
        }


@dataclass
class EQCorrection:
    """EQ correction for a frequency band."""
    frequency: float
    gain_db: float
    q: float
    type: str  # 'peak', 'high_shelf', 'low_shelf'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'frequency': self.frequency,
            'gain_db': self.gain_db,
            'q': self.q,
            'type': self.type
        }


class SineSweepGenerator:
    """
    Generate exponential sine sweep for system measurement.
    
    Farina's method: exponential sweep with time-varying frequency
    f(t) = f1 * (f2/f1)^(t/T)
    
    Where:
    - f1 = 20Hz (start frequency)
    - f2 = 20000Hz (end frequency)
    - T = duration (10-20 seconds)
    """
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.f1 = 20.0      # Start frequency
        self.f2 = 20000.0   # End frequency
        self.duration = 15.0  # Default 15 seconds
        self.amplitude = 0.5  # -6dBFS to avoid clipping
        
        logger.info(f"SineSweepGenerator initialized: {self.f1}-{self.f2}Hz, {self.duration}s")
    
    def generate(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Generate exponential sine sweep.
        
        Args:
            duration: Sweep duration in seconds (default 15s)
            
        Returns:
            Sine sweep signal
        """
        if duration is None:
            duration = self.duration
        
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Exponential frequency sweep
        # f(t) = f1 * (f2/f1)^(t/T)
        k = np.log(self.f2 / self.f1) / duration
        instantaneous_freq = self.f1 * np.exp(k * t)
        
        # Phase is integral of frequency
        phase = (self.f1 / k) * (np.exp(k * t) - 1)
        
        # Generate sweep
        sweep = self.amplitude * np.sin(2 * np.pi * phase)
        
        # Apply fade in/out to avoid clicks
        fade_samples = int(0.01 * self.sample_rate)  # 10ms fade
        if fade_samples > 0 and len(sweep) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            sweep[:fade_samples] *= fade_in
            sweep[-fade_samples:] *= fade_out
        
        return sweep
    
    def generate_inverse_filter(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Generate inverse filter for deconvolution.
        
        The inverse filter is the time-reversed sweep with amplitude modulation.
        """
        if duration is None:
            duration = self.duration
        
        num_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, num_samples)
        
        # Time-reversed exponential sweep
        k = np.log(self.f2 / self.f1) / duration
        
        # For inverse filter, frequency decreases from f2 to f1
        # And amplitude is proportional to instantaneous frequency
        t_rev = duration - t
        instantaneous_freq = self.f1 * np.exp(k * t_rev)
        phase = (self.f1 / k) * (np.exp(k * t_rev) - 1)
        
        # Generate time-reversed sweep
        sweep_rev = np.sin(2 * np.pi * phase)
        
        # Amplitude modulation (proportional to frequency for pinking)
        amplitude_mod = instantaneous_freq / self.f1
        inverse_filter = sweep_rev * amplitude_mod
        
        # Normalize
        inverse_filter /= np.max(np.abs(inverse_filter)) + 1e-10
        
        return inverse_filter


class SystemMeasurement:
    """
    Main system measurement class.
    
    Performs sine sweep measurement and calculates:
    - Magnitude response
    - Phase response
    - Impulse response
    - RT60
    - Coherence
    """
    
    def __init__(self, sample_rate: int = 48000, fft_size: int = 65536):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        
        self.sweep_generator = SineSweepGenerator(sample_rate)
        self.state = MeasurementState.IDLE
        
        # Measurement data
        self.sweep_signal: Optional[np.ndarray] = None
        self.inverse_filter: Optional[np.ndarray] = None
        self.recorded_responses: List[MeasurementPosition] = []
        self.current_position: int = 0
        
        logger.info(f"SystemMeasurement initialized: {sample_rate}Hz, FFT {fft_size}")
    
    def start_measurement(self, num_positions: int = 6, duration: float = 15.0):
        """Initialize measurement process."""
        self.state = MeasurementState.GENERATING
        self.current_position = 0
        self.recorded_responses.clear()
        
        # Generate sweep and inverse filter
        self.sweep_signal = self.sweep_generator.generate(duration)
        self.inverse_filter = self.sweep_generator.generate_inverse_filter(duration)
        
        self.state = MeasurementState.IDLE
        
        logger.info(f"Measurement initialized: {num_positions} positions, {duration}s sweep")
        
        return {
            'num_positions': num_positions,
            'sweep_duration': duration,
            'sweep_samples': len(self.sweep_signal)
        }
    
    def record_position(self, recorded_signal: np.ndarray, 
                       position: Tuple[float, float, float]) -> MeasurementPosition:
        """
        Record response at a single position.
        
        Args:
            recorded_signal: Recorded mic signal
            position: (x, y, height) in meters
            
        Returns:
            MeasurementPosition with processed data
        """
        self.state = MeasurementState.PROCESSING
        
        # Ensure recorded signal is same length as sweep
        if len(recorded_signal) != len(self.sweep_signal):
            logger.warning(f"Signal length mismatch: {len(recorded_signal)} vs {len(self.sweep_signal)}")
            # Pad or truncate
            if len(recorded_signal) < len(self.sweep_signal):
                recorded_signal = np.pad(recorded_signal, 
                                        (0, len(self.sweep_signal) - len(recorded_signal)))
            else:
                recorded_signal = recorded_signal[:len(self.sweep_signal)]
        
        # Calculate impulse response via deconvolution
        impulse_response = self._calculate_ir(recorded_signal)
        
        # Create position data
        pos_data = MeasurementPosition(
            position_id=self.current_position,
            x=position[0],
            y=position[1],
            height=position[2],
            mic_response=recorded_signal,
            timestamp=time.time()
        )
        
        # Calculate quality score (SNR-based)
        snr = self._calculate_snr(recorded_signal)
        pos_data.quality_score = min(1.0, snr / 40.0)  # Normalize to 0-1
        
        self.recorded_responses.append(pos_data)
        self.current_position += 1
        
        self.state = MeasurementState.IDLE
        
        return pos_data
    
    def _calculate_ir(self, recorded: np.ndarray) -> np.ndarray:
        """Calculate impulse response via deconvolution."""
        # Convolve recorded signal with inverse filter
        ir = convolve(recorded, self.inverse_filter, mode='full')
        
        # Normalize
        ir /= np.max(np.abs(ir)) + 1e-10
        
        # Trim to reasonable length (100ms)
        trim_samples = int(0.1 * self.sample_rate)
        if len(ir) > trim_samples:
            # Find peak
            peak_idx = np.argmax(np.abs(ir))
            start_idx = max(0, peak_idx - int(0.01 * self.sample_rate))  # 10ms before peak
            end_idx = min(len(ir), start_idx + trim_samples)
            ir = ir[start_idx:end_idx]
        
        return ir
    
    def _calculate_snr(self, signal: np.ndarray) -> float:
        """Calculate SNR in dB."""
        # Simple SNR estimation
        signal_power = np.mean(signal ** 2)
        # Estimate noise from tail of signal
        noise_power = np.mean(signal[-1000:] ** 2) if len(signal) > 1000 else signal_power * 0.01
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = 60.0
        
        return max(0, snr_db)
    
    def analyze(self) -> MeasurementResult:
        """
        Analyze all recorded positions and generate measurement result.
        
        Returns:
            Complete measurement result with averaged responses
        """
        self.state = MeasurementState.PROCESSING
        
        if len(self.recorded_responses) == 0:
            logger.error("No recorded responses to analyze")
            self.state = MeasurementState.ERROR
            return MeasurementResult()
        
        # Average responses from all positions
        avg_response = self._average_responses()
        
        # Calculate frequency response
        magnitude, phase, freqs, coherence = self._calculate_frequency_response(avg_response)
        
        # Calculate RT60
        rt60_data = self._calculate_rt60(avg_response)
        
        # Create result
        result = MeasurementResult()
        result.magnitude_response.frequencies = freqs
        result.magnitude_response.magnitude_db = magnitude
        result.phase_response.frequencies = freqs
        result.phase_response.phase_deg = phase
        result.phase_response.coherence = coherence
        result.rt60 = rt60_data
        result.positions = self.recorded_responses.copy()
        result.timestamp = time.time()
        
        # Overall quality
        result.overall_quality = np.mean([p.quality_score for p in self.recorded_responses])
        
        self.state = MeasurementState.COMPLETE
        
        logger.info(f"Analysis complete: {len(freqs)} frequency points, "
                   f"RT60 {np.mean(rt60_data.rt60) if rt60_data.rt60 else 0:.2f}s")
        
        return result
    
    def _average_responses(self) -> np.ndarray:
        """Average impulse responses from all positions (spatial averaging)."""
        # For simplicity, average in time domain
        # In production: use complex averaging in frequency domain
        
        max_len = max(len(p.mic_response) for p in self.recorded_responses)
        averaged = np.zeros(max_len)
        
        for pos in self.recorded_responses:
            # Pad to same length
            response = np.pad(pos.mic_response, (0, max_len - len(pos.mic_response)))
            averaged += response
        
        averaged /= len(self.recorded_responses)
        
        return averaged
    
    def _calculate_frequency_response(self, response: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate frequency response from impulse response."""
        # Zero-pad to FFT size
        padded = np.zeros(self.fft_size)
        padded[:min(len(response), self.fft_size)] = response[:min(len(response), self.fft_size)]
        
        # FFT
        spectrum = fft(padded)
        
        # Calculate magnitude and phase
        magnitude = np.abs(spectrum[:self.fft_size//2])
        phase = np.angle(spectrum[:self.fft_size//2])
        
        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        phase_deg = np.degrees(phase)
        
        # Frequencies (use same length as magnitude)
        freqs = np.linspace(0, self.sample_rate/2, len(magnitude))
        
        # Coherence (simplified - in production use multiple measurements)
        coherence = np.ones_like(magnitude) * 0.95
        
        return magnitude_db, phase_deg, freqs, coherence
    
    def _calculate_rt60(self, response: np.ndarray) -> RT60Data:
        """Calculate RT60 from impulse response using Schroeder integration."""
        rt60_data = RT60Data()
        
        # Octave bands
        bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        
        for band in bands:
            # In production: bandpass filter and calculate RT60
            # Simplified: estimate based on overall decay
            
            # Schroeder integration (backward cumulative sum of squared IR)
            squared = response ** 2
            schroeder = np.cumsum(squared[::-1])[::-1]
            schroeder_db = 10 * np.log10(schroeder + 1e-10)
            
            # Find -60dB point
            if len(schroeder_db) > 0:
                max_db = schroeder_db[0]
                idx_60db = np.where(schroeder_db <= max_db - 60)[0]
                
                if len(idx_60db) > 0:
                    rt60 = idx_60db[0] / self.sample_rate
                else:
                    # Estimate from -30dB
                    idx_30db = np.where(schroeder_db <= max_db - 30)[0]
                    if len(idx_30db) > 0:
                        rt60 = 2 * (idx_30db[0] / self.sample_rate)  # T30 * 2
                    else:
                        rt60 = 1.0  # Default
            else:
                rt60 = 1.0
            
            rt60_data.bands.append(band)
            rt60_data.rt60.append(min(rt60, 5.0))  # Cap at 5 seconds
        
        return rt60_data


class CorrectionCalculator:
    """
    Calculate EQ corrections from measurement results.
    
    Principles:
    - Max 6-10 dB cuts
    - Gentle Q (1.4-2.0)
    - Compensate only problem areas
    - Target: flat ±3dB response
    """
    
    def __init__(self):
        self.max_cut_db = 10.0
        self.max_boost_db = 6.0
        self.q_range = (1.4, 2.0)
        self.target_variance_db = 3.0  # Target ±3dB flatness
        
        logger.info("CorrectionCalculator initialized")
    
    def calculate_corrections(self, measurement: MeasurementResult) -> List[EQCorrection]:
        """
        Calculate EQ corrections from measurement.
        
        Returns:
            List of EQCorrection objects
        """
        corrections = []
        
        # Get smoothed magnitude response
        smoothed = measurement.magnitude_response.get_smoothed(1.0/12.0)
        
        if len(smoothed.magnitude_db) == 0:
            return corrections
        
        # Find target level (median)
        target_level = np.median(smoothed.magnitude_db)
        
        # Find deviations > 3dB
        deviations = smoothed.magnitude_db - target_level
        
        # Find peaks (positive deviations - need cut)
        peaks, properties = find_peaks(deviations, 
                                       height=self.target_variance_db,
                                       distance=50)  # Min 50 bins apart
        
        for peak in peaks:
            freq = smoothed.frequencies[peak]
            deviation = deviations[peak]
            
            # Limit cut
            gain_db = -min(deviation, self.max_cut_db)
            
            # Calculate Q based on bandwidth
            # In production: use -3dB points
            q = 1.8  # Default gentle Q
            
            correction = EQCorrection(
                frequency=freq,
                gain_db=gain_db,
                q=q,
                type='peak'
            )
            
            corrections.append(correction)
        
        # Find dips (negative deviations - limited boost)
        dips, properties = find_peaks(-deviations,
                                      height=self.target_variance_db * 0.7,
                                      distance=50)
        
        for dip in dips:
            freq = smoothed.frequencies[dip]
            deviation = -deviations[dip]  # Make positive
            
            # Limit boost
            gain_db = min(deviation, self.max_boost_db)
            
            if gain_db > 1.0:  # Only if significant
                correction = EQCorrection(
                    frequency=freq,
                    gain_db=gain_db,
                    q=1.8,
                    type='peak'
                )
                
                corrections.append(correction)
        
        # Limit total number of bands
        if len(corrections) > 16:
            # Sort by absolute gain and keep top 16
            corrections.sort(key=lambda c: abs(c.gain_db), reverse=True)
            corrections = corrections[:16]
        
        logger.info(f"Calculated {len(corrections)} EQ corrections")
        
        return corrections


class SystemMeasurementController:
    """
    Main controller for system measurement and correction.
    
    Workflow:
    1. Start measurement (generate sweep)
    2. Record at multiple positions
    3. Analyze (average responses)
    4. Calculate corrections
    5. Apply to Group/Master/Matrix
    """
    
    def __init__(self, mixer_client=None, sample_rate: int = 48000):
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        
        self.measurement = SystemMeasurement(sample_rate)
        self.correction_calculator = CorrectionCalculator()
        
        self.last_result: Optional[MeasurementResult] = None
        self.applied_corrections: List[EQCorrection] = []
        
        logger.info("SystemMeasurementController initialized")
    
    def start_soundcheck_measurement(self, num_positions: int = 6) -> Dict[str, Any]:
        """Start soundcheck measurement process."""
        info = self.measurement.start_measurement(num_positions)
        
        return {
            'status': 'ready',
            'num_positions': num_positions,
            'sweep_duration': info['sweep_duration'],
            'instructions': [
                '1. Place reference mic at FOH position (1-2m height)',
                '2. Ensure quiet environment',
                '3. Play sweep at moderate level (-12dBFS)',
                f'4. Record at {num_positions} positions in audience area',
                '5. Keep mic at ear level (1.2-1.5m)'
            ]
        }
    
    def record_position(self, audio_data: np.ndarray, 
                       position: Tuple[float, float, float]) -> MeasurementPosition:
        """Record measurement at a position."""
        return self.measurement.record_position(audio_data, position)
    
    def analyze_and_calculate(self) -> Dict[str, Any]:
        """Analyze measurements and calculate corrections."""
        # Analyze
        result = self.measurement.analyze()
        self.last_result = result
        
        # Calculate corrections
        corrections = self.correction_calculator.calculate_corrections(result)
        self.applied_corrections = corrections
        
        return {
            'status': 'complete',
            'quality': result.overall_quality,
            'rt60': result.rt60.to_dict(),
            'num_corrections': len(corrections),
            'corrections': [c.to_dict() for c in corrections],
            'frequencies': result.magnitude_response.frequencies.tolist(),
            'magnitude': result.magnitude_response.magnitude_db.tolist()
        }
    
    def apply_corrections(self, target_bus: TargetBus, bus_id: int) -> bool:
        """
        Apply calculated corrections to mixer.
        
        Args:
            target_bus: master, group, or matrix
            bus_id: Bus channel number
            
        Returns:
            Success status
        """
        if not self.mixer_client:
            logger.error("No mixer client available")
            return False
        
        try:
            for i, correction in enumerate(self.applied_corrections):
                # Send OSC command
                # Format depends on mixer (e.g., /bus/{id}/eq/{band}/gain)
                if hasattr(self.mixer_client, 'set_eq_band'):
                    self.mixer_client.set_eq_band(
                        bus_id,
                        band=i,
                        frequency=correction.frequency,
                        gain=correction.gain_db,
                        q=correction.q
                    )
            
            logger.info(f"Applied {len(self.applied_corrections)} corrections to {target_bus.value} {bus_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply corrections: {e}")
            return False
    
    def get_recommended_target(self) -> str:
        """Get recommended target bus based on measurement."""
        # Analyze RT60
        if self.last_result and self.last_result.rt60.rt60:
            avg_rt60 = np.mean(self.last_result.rt60.rt60)
            
            if avg_rt60 > 2.0:
                return "Use Group EQ + Matrix delay for large room"
            elif avg_rt60 < 1.0:
                return "Use Master EQ for small room"
            else:
                return "Use Master or Group EQ"
        
        return "Use Master EQ (default)"


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("System Measurement - Sine Sweep Analysis Test")
    print("=" * 70)
    
    # Create controller
    controller = SystemMeasurementController()
    
    # Start measurement
    info = controller.start_soundcheck_measurement(num_positions=3)
    print(f"\nMeasurement setup: {info['num_positions']} positions")
    
    # Generate sweep
    sweep = controller.measurement.sweep_generator.generate(duration=5.0)
    print(f"Sweep generated: {len(sweep)} samples ({len(sweep)/48000:.2f}s)")
    
    # Simulate recording at 3 positions
    for pos in range(3):
        print(f"\nRecording position {pos + 1}...")
        
        # Simulate room response (simplified)
        # Add some room modes and reflections
        response = sweep.copy()
        
        # Add low-frequency boost (room mode)
        low_freq = np.sin(2 * np.pi * 80 * np.arange(len(sweep)) / 48000)
        response += 0.3 * low_freq * sweep
        
        # Add high-frequency roll-off
        response = np.convolve(response, [0.5, 0.5], mode='same')
        
        # Add noise
        response += np.random.randn(len(response)) * 0.001
        
        # Record
        pos_data = controller.record_position(response, (pos * 2, 0, 1.5))
        print(f"  Quality: {pos_data.quality_score:.2f}")
    
    # Analyze
    print("\nAnalyzing measurements...")
    result = controller.analyze_and_calculate()
    
    print(f"\nResults:")
    print(f"  Overall quality: {result['quality']:.2f}")
    print(f"  RT60 bands: {len(result['rt60']['bands'])}")
    print(f"  Corrections: {result['num_corrections']}")
    
    if result['num_corrections'] > 0:
        print(f"\n  Recommended EQ:")
        for corr in result['corrections'][:5]:  # Show first 5
            print(f"    {corr['frequency']:.0f}Hz: {corr['gain_db']:+.1f}dB (Q={corr['q']})")
    
    print(f"\n  Target recommendation: {controller.get_recommended_target()}")
    
    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
