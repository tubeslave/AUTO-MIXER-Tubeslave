"""
Auto Phase Alignment Module - GCC-PHAT Implementation

Based on Intelligent Music Production (De Man, Stables, Reiss)
Section 8.2.3: Time Alignment using GCC-PHAT

Method:
1. Compute GCC-PHAT between reference and target channel
2. Find delay from peak of inverse FFT
3. Apply parabolic interpolation for sub-sample accuracy
4. Send OSC delay command to mixer
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from scipy.fft import fft, ifft
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DelayMeasurement:
    """Result of delay measurement between two channels."""
    delay_ms: float          # Estimated delay in milliseconds
    delay_samples: float     # Estimated delay in samples (sub-sample precision)
    correlation_peak: float  # Peak correlation value (0-1)
    psr: float              # Peak-to-Sidelobe Ratio
    snr_db: float           # Signal-to-Noise Ratio in dB
    confidence: float       # Overall confidence (0-1)
    coherence: float        # Magnitude squared coherence
    
    def is_valid(self, min_correlation: float = 0.5, min_psr: float = 5.0) -> bool:
        """Check if measurement is valid based on thresholds."""
        return (self.correlation_peak >= min_correlation and 
                self.psr >= min_psr and
                self.confidence > 0.6)


class GCCPHATAnalyzer:
    """
    GCC-PHAT analyzer for time delay estimation.
    
    Based on:
    - Knapp and Carter (1976) "The Generalized Correlation Method for Estimation of Time Delay"
    - Section 8.2.3 from "Intelligent Music Production" (De Man et al.)
    """
    
    def __init__(self, 
                 sample_rate: int = 48000,
                 fft_size: int = 4096,
                 hop_size: int = 2048,
                 max_delay_ms: float = 50.0):
        """
        Args:
            sample_rate: Audio sample rate in Hz
            fft_size: FFT size for analysis (larger = better frequency resolution)
            hop_size: Hop size between frames (overlap = fft_size - hop_size)
            max_delay_ms: Maximum delay to search for (ms)
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000.0)
        
        # Pre-compute window
        self.window = np.hanning(fft_size)
        
        # Circular buffer for audio
        self.buffer_size = fft_size * 8  # Large enough for test signals
        self.ref_buffer = np.zeros(self.buffer_size)
        self.tgt_buffer = np.zeros(self.buffer_size)
        self.buffer_idx = 0
        
        # Temporal smoothing
        self.delay_history: deque = deque(maxlen=10)
        self.confidence_history: deque = deque(maxlen=10)
        
        logger.info(f"GCC-PHAT Analyzer initialized: {fft_size} FFT, {sample_rate}Hz")
    
    def add_frames(self, ref_frame: np.ndarray, tgt_frame: np.ndarray):
        """Add new audio frames to circular buffer."""
        frame_size = len(ref_frame)
        
        # Add to circular buffer
        end_idx = self.buffer_idx + frame_size
        if end_idx <= self.buffer_size:
            self.ref_buffer[self.buffer_idx:end_idx] = ref_frame
            self.tgt_buffer[self.buffer_idx:end_idx] = tgt_frame
        else:
            # Wrap around
            first_part = self.buffer_size - self.buffer_idx
            self.ref_buffer[self.buffer_idx:] = ref_frame[:first_part]
            self.tgt_buffer[self.buffer_idx:] = tgt_frame[:first_part]
            second_part = end_idx - self.buffer_size
            self.ref_buffer[:second_part] = ref_frame[first_part:first_part + second_part]
            self.tgt_buffer[:second_part] = tgt_frame[first_part:first_part + second_part]
        
        self.buffer_idx = end_idx % self.buffer_size
    
    def compute_delay(self, 
                     ref_signal: Optional[np.ndarray] = None,
                     tgt_signal: Optional[np.ndarray] = None) -> DelayMeasurement:
        """
        Compute delay between reference and target signals using GCC-PHAT.
        
        Args:
            ref_signal: Reference signal (if None, uses buffer)
            tgt_signal: Target signal (if None, uses buffer)
            
        Returns:
            DelayMeasurement with estimated delay and quality metrics
        """
        # Use buffer if signals not provided
        if ref_signal is None:
            ref_signal = self.ref_buffer
        if tgt_signal is None:
            tgt_signal = self.tgt_buffer
        
        # Ensure we have enough samples
        if len(ref_signal) < self.fft_size:
            logger.warning(f"Signal too short: {len(ref_signal)} < {self.fft_size}")
            return DelayMeasurement(0, 0, 0, 0, -60, 0, 0)
        
        # Extract frames with windowing
        ref_frame = ref_signal[:self.fft_size] * self.window
        tgt_frame = tgt_signal[:self.fft_size] * self.window
        
        # Compute FFT
        Ref = fft(ref_frame)
        Tgt = fft(tgt_frame)
        
        # Compute Cross-Power Spectrum
        CrossSpectrum = Ref.conj() * Tgt
        
        # PHAT normalization (whitening)
        # This makes the method robust to spectral coloring
        eps = 1e-10  # Prevent division by zero
        PHAT = CrossSpectrum / (np.abs(CrossSpectrum) + eps)
        
        # Inverse FFT to get GCC
        gcc = ifft(PHAT).real
        
        # Circular shift to center zero delay
        gcc = np.fft.fftshift(gcc)
        
        # Limit search range to max_delay_samples
        center = len(gcc) // 2
        search_start = max(0, center - self.max_delay_samples)
        search_end = min(len(gcc), center + self.max_delay_samples + 1)
        gcc_limited = gcc[search_start:search_end]
        
        # Find peak
        peak_idx = np.argmax(np.abs(gcc_limited))
        peak_value = gcc_limited[peak_idx]
        
        # Parabolic interpolation for sub-sample precision
        if 0 < peak_idx < len(gcc_limited) - 1:
            alpha = gcc_limited[peak_idx - 1]
            beta = gcc_limited[peak_idx]
            gamma = gcc_limited[peak_idx + 1]
            
            # Parabolic interpolation formula
            p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
            interpolated_peak_idx = peak_idx + p
            interpolated_peak_value = beta - 0.25 * (alpha - gamma) * p
        else:
            interpolated_peak_idx = peak_idx
            interpolated_peak_value = peak_value
        
        # Convert to delay in samples
        delay_samples = interpolated_peak_idx - (len(gcc_limited) // 2)
        delay_ms = delay_samples * 1000.0 / self.sample_rate
        
        # Compute quality metrics
        # Correlation coefficient
        correlation_peak = np.abs(interpolated_peak_value) / np.sqrt(
            np.sum(ref_frame**2) * np.sum(tgt_frame**2)
        )
        
        # Peak-to-Sidelobe Ratio (PSR)
        # Find second highest peak (outside immediate vicinity)
        vicinity = 5
        gcc_excluded = gcc_limited.copy()
        start_excl = max(0, peak_idx - vicinity)
        end_excl = min(len(gcc_excluded), peak_idx + vicinity + 1)
        gcc_excluded[start_excl:end_excl] = 0
        second_peak = np.max(np.abs(gcc_excluded))
        psr = 20 * np.log10(np.abs(interpolated_peak_value) / (second_peak + eps))
        
        # SNR estimation
        signal_power = np.mean(ref_frame**2)
        # Estimate noise from correlation floor
        noise_floor = np.mean(np.abs(gcc_limited)**2)
        snr_db = 10 * np.log10(signal_power / (noise_floor + eps))
        
        # Coherence (magnitude squared)
        coherence = np.abs(CrossSpectrum)**2 / (
            (np.abs(Ref)**2 + eps) * (np.abs(Tgt)**2 + eps)
        )
        coherence = np.mean(coherence[:self.fft_size//2])
        
        # Overall confidence
        confidence = (
            0.4 * min(1.0, correlation_peak / 0.8) +
            0.3 * min(1.0, psr / 10.0) +
            0.2 * min(1.0, max(0, snr_db) / 40.0) +
            0.1 * coherence
        )
        
        # Temporal smoothing
        self.delay_history.append(delay_samples)
        self.confidence_history.append(confidence)
        
        # Use median of recent measurements if confidence is high enough
        if len(self.delay_history) >= 3 and np.mean(self.confidence_history) > 0.5:
            smoothed_delay = np.median(self.delay_history)
            # Blend current and smoothed based on confidence
            alpha = min(1.0, confidence * 1.5)
            delay_samples = alpha * delay_samples + (1 - alpha) * smoothed_delay
            delay_ms = delay_samples * 1000.0 / self.sample_rate
        
        return DelayMeasurement(
            delay_ms=delay_ms,
            delay_samples=delay_samples,
            correlation_peak=correlation_peak,
            psr=psr,
            snr_db=snr_db,
            confidence=confidence,
            coherence=coherence
        )
    
    def reset(self):
        """Reset analyzer state."""
        self.ref_buffer.fill(0)
        self.tgt_buffer.fill(0)
        self.buffer_idx = 0
        self.delay_history.clear()
        self.confidence_history.clear()


class AutoPhaseAligner:
    """
    Automatic phase alignment controller.
    
    Manages delay estimation and OSC command sending for multiple channels.
    """
    
    def __init__(self,
                 mixer_client=None,
                 sample_rate: int = 48000,
                 fft_size: int = 4096):
        """
        Args:
            mixer_client: OSC mixer client
            sample_rate: Audio sample rate
            fft_size: FFT size for GCC-PHAT
        """
        self.mixer_client = mixer_client
        self.sample_rate = sample_rate
        
        # Create analyzer
        self.analyzer = GCCPHATAnalyzer(
            sample_rate=sample_rate,
            fft_size=fft_size
        )
        
        # Channel management
        self.reference_channel: Optional[int] = None
        self.channels_to_align: List[int] = []
        self.channel_analyzers: Dict[int, GCCPHATAnalyzer] = {}
        
        # State
        self.is_running = False
        self.measurements: Dict[int, DelayMeasurement] = {}
        self.last_update_time: Dict[int, float] = {}
        
        # Settings
        self.min_update_interval_ms = 100  # Rate limiting
        self.correlation_threshold = 0.5
        self.psr_threshold = 5.0
        
        logger.info("AutoPhaseAligner initialized")
    
    def set_reference_channel(self, channel: int):
        """Set the reference channel."""
        self.reference_channel = channel
        logger.info(f"Reference channel set to {channel}")
    
    def add_channel(self, channel: int):
        """Add a channel to align."""
        if channel not in self.channels_to_align:
            self.channels_to_align.append(channel)
            self.channel_analyzers[channel] = GCCPHATAnalyzer(
                sample_rate=self.sample_rate
            )
            logger.info(f"Added channel {channel} for alignment")
    
    def remove_channel(self, channel: int):
        """Remove a channel from alignment."""
        if channel in self.channels_to_align:
            self.channels_to_align.remove(channel)
            del self.channel_analyzers[channel]
            logger.info(f"Removed channel {channel}")
    
    def process_audio(self, channel: int, ref_audio: np.ndarray, tgt_audio: np.ndarray):
        """
        Process audio frames and compute delay.
        
        Args:
            channel: Target channel ID
            ref_audio: Reference channel audio
            tgt_audio: Target channel audio
        """
        if channel not in self.channel_analyzers:
            return
        
        analyzer = self.channel_analyzers[channel]
        analyzer.add_frames(ref_audio, tgt_audio)
        
        # Rate limiting
        current_time = time.time() * 1000
        if channel in self.last_update_time:
            if current_time - self.last_update_time[channel] < self.min_update_interval_ms:
                return
        
        # Compute delay
        measurement = analyzer.compute_delay()
        self.measurements[channel] = measurement
        
        # Check if valid
        if measurement.is_valid(self.correlation_threshold, self.psr_threshold):
            # Send OSC command if mixer client available
            if self.mixer_client and abs(measurement.delay_ms) > 0.1:
                self._send_delay_command(channel, measurement)
        
        self.last_update_time[channel] = current_time
    
    def _send_delay_command(self, channel: int, measurement: DelayMeasurement):
        """Send OSC delay command to mixer."""
        try:
            # Round to mixer precision (typically 0.02ms @ 48kHz)
            delay_ms = round(measurement.delay_ms * 50) / 50  # 0.02ms resolution
            
            # Only apply if delay is positive (add delay to faster channel)
            if delay_ms > 0:
                self.mixer_client.set_channel_delay(channel, delay_ms)
                logger.debug(f"Set delay for channel {channel}: {delay_ms:.2f}ms "
                           f"(confidence: {measurement.confidence:.2f})")
        except Exception as e:
            logger.error(f"Failed to send delay command: {e}")
    
    def get_status(self) -> Dict:
        """Get current alignment status."""
        return {
            'reference_channel': self.reference_channel,
            'channels': self.channels_to_align,
            'measurements': {
                ch: {
                    'delay_ms': m.delay_ms,
                    'confidence': m.confidence,
                    'correlation': m.correlation_peak,
                    'valid': m.is_valid()
                }
                for ch, m in self.measurements.items()
            }
        }
    
    def reset(self):
        """Reset all analyzers."""
        self.analyzer.reset()
        for analyzer in self.channel_analyzers.values():
            analyzer.reset()
        self.measurements.clear()
        self.last_update_time.clear()


import time


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("GCC-PHAT Auto Phase Alignment Test")
    print("=" * 60)
    
    # Test parameters
    sample_rate = 48000
    test_delays = [0, 10, 25, 50, 100]  # ms
    
    for true_delay_ms in test_delays:
        print(f"\nTest: True delay = {true_delay_ms} ms")
        
        # Create test signals
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Reference: pink noise
        ref_signal = np.random.randn(len(t))
        ref_signal = np.convolve(ref_signal, [0.5, 0.5], mode='same')  # Simple pink
        
        # Target: delayed version
        delay_samples = int(true_delay_ms * sample_rate / 1000)
        tgt_signal = np.zeros_like(ref_signal)
        if delay_samples > 0:
            tgt_signal[delay_samples:] = ref_signal[:-delay_samples]
        else:
            tgt_signal = ref_signal
        
        # Analyze
        analyzer = GCCPHATAnalyzer(sample_rate=sample_rate)
        analyzer.add_frames(ref_signal, tgt_signal)
        measurement = analyzer.compute_delay()
        
        print(f"  Estimated delay: {measurement.delay_ms:.2f} ms "
              f"({measurement.delay_samples:.1f} samples)")
        print(f"  Error: {abs(measurement.delay_ms - true_delay_ms):.3f} ms")
        print(f"  Correlation: {measurement.correlation_peak:.3f}")
        print(f"  PSR: {measurement.psr:.1f} dB")
        print(f"  Confidence: {measurement.confidence:.2f}")
        print(f"  Valid: {'✓' if measurement.is_valid() else '✗'}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
