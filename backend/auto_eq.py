"""
Auto-EQ Module with Spectrum Analysis and Correction

Модуль автоматической эквализации с анализом спектра (Essentia),
профилями инструментов, алгоритмами коррекции (SciPy),
локальным превью (Pedalboard) и отправкой команд на микшер (python-osc).
"""

import numpy as np
import threading
import logging
import time
from typing import Dict, List, Callable, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# SciPy for signal processing
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EQBand:
    """Represents a single EQ band with frequency, gain, and Q parameters."""
    band_type: str  # 'low_shelf', 'peak', 'high_shelf'
    frequency: float  # Hz
    gain: float  # dB (-15 to +15)
    q: float  # Q factor (0.44 to 10)
    
    def to_dict(self) -> dict:
        return {
            'band_type': self.band_type,
            'frequency': self.frequency,
            'gain': round(self.gain, 2),
            'q': round(self.q, 2)
        }


@dataclass
class SpectralData:
    """Contains spectral analysis results."""
    spectrum: np.ndarray  # Magnitude spectrum
    frequencies: np.ndarray  # Frequency bins
    peak_freq: float  # Peak frequency in Hz
    centroid: float  # Spectral centroid in Hz
    rolloff: float  # Spectral rolloff in Hz
    flatness: float  # Spectral flatness (0-1)
    bandwidth: float  # Spectral bandwidth in Hz
    peaks: List[Tuple[float, float]]  # List of (frequency, magnitude) peaks
    
    def to_dict(self) -> dict:
        return {
            'peak_freq': round(self.peak_freq, 1),
            'centroid': round(self.centroid, 1),
            'rolloff': round(self.rolloff, 1),
            'flatness': round(self.flatness, 4),
            'bandwidth': round(self.bandwidth, 1),
            'peaks': [(round(f, 1), round(m, 2)) for f, m in self.peaks[:10]]
        }


@dataclass
class InstrumentProfile:
    """Target EQ profile for a specific instrument type."""
    name: str
    description: str
    # Target curve as list of (frequency, target_db) points
    target_curve: List[Tuple[float, float]]
    # Problem frequencies to cut
    cut_frequencies: List[Tuple[float, float, float]]  # (freq, max_cut_db, q)
    # Frequencies to boost
    boost_frequencies: List[Tuple[float, float, float]]  # (freq, max_boost_db, q)
    # Low cut recommendation
    low_cut_freq: Optional[float] = None
    # High cut recommendation
    high_cut_freq: Optional[float] = None


# ============================================================================
# Spectrum Analyzer (Essentia)
# ============================================================================

class SpectrumAnalyzer:
    """
    Real-time spectrum analyzer using Essentia library.
    
    Provides FFT-based spectral analysis with features like:
    - Magnitude spectrum
    - Spectral peaks detection
    - Spectral centroid, rolloff, flatness
    - MFCC for timbre analysis
    """
    
    def __init__(self,
                 sample_rate: int = 48000,
                 frame_size: int = 4096,
                 hop_size: int = 2048,
                 device_index: int = None,
                 audio_capture=None):
        """
        Initialize spectrum analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: FFT frame size (power of 2)
            hop_size: Hop size between frames
            device_index: PyAudio device index for audio input
            audio_capture: Optional unified AudioCapture service
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.device_index = device_index
        self._audio_capture = audio_capture
        
        # Essentia algorithms (lazy initialization)
        self._essentia_initialized = False
        self._essentia_error_logged = False  # Track if error already logged
        self._windowing = None
        self._spectrum = None
        self._spectral_peaks = None
        self._spectral_centroid = None
        self._spectral_flatness = None
        self._rolloff = None
        self._mfcc = None
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=int(sample_rate * 2))  # 2 seconds buffer
        
        # Current spectral data
        self.current_spectrum: Optional[SpectralData] = None
        
        # Frequency bins for FFT
        self.freq_bins = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
        
        # State
        self.is_running = False
        self._stop_event = threading.Event()
        self._analysis_thread = None
        
        # PyAudio
        self.pa = None
        self.stream = None
        self._num_channels = 1
        
        # Callbacks
        self.on_spectrum_updated: Optional[Callable[[SpectralData], None]] = None
        
        logger.info(f"SpectrumAnalyzer initialized: {sample_rate}Hz, frame_size={frame_size}")
    
    def _init_essentia(self):
        """Initialize Essentia algorithms (lazy loading)."""
        if self._essentia_initialized:
            return True
        
        # If already tried and failed, don't try again
        if self._essentia_error_logged and not self._essentia_initialized:
            return False
        
        try:
            import essentia.standard as es
            
            self._windowing = es.Windowing(type='hann', size=self.frame_size)
            self._spectrum = es.Spectrum(size=self.frame_size)
            self._spectral_peaks = es.SpectralPeaks(
                sampleRate=self.sample_rate,
                maxPeaks=20,
                magnitudeThreshold=0.00001,
                minFrequency=20,
                maxFrequency=20000,
                orderBy='magnitude'
            )
            self._spectral_centroid = es.Centroid(range=self.sample_rate / 2)
            self._spectral_flatness = es.Flatness()
            self._rolloff = es.RollOff(sampleRate=self.sample_rate)
            self._mfcc = es.MFCC(
                inputSize=self.frame_size // 2 + 1,
                sampleRate=self.sample_rate,
                numberCoefficients=13
            )
            
            self._essentia_initialized = True
            logger.info("Essentia algorithms initialized successfully")
            return True
            
        except ImportError as e:
            if not self._essentia_error_logged:
                logger.warning(f"Essentia not installed: {e}. Falling back to numpy-based spectrum analysis.")
                self._essentia_error_logged = True
            self._essentia_initialized = False
            return False
        except Exception as e:
            if not self._essentia_error_logged:
                logger.warning(f"Error initializing Essentia: {e}. Falling back to numpy-based spectrum analysis.")
                self._essentia_error_logged = True
            self._essentia_initialized = False
            return False
    
    def _audio_capture_poll(self):
        """Poll AudioCapture buffer for the target channel."""
        if not self._audio_capture:
            return
        ch = self._target_channel + 1  # AudioCapture uses 1-based channels
        data = self._audio_capture.get_buffer(ch, self.hop_size)
        if data is not None and len(data) > 0:
            self.audio_buffer.extend(data)

    def start(self, channel: int = 1, on_spectrum_callback: Callable = None) -> bool:
        """
        Start spectrum analysis.

        Args:
            channel: Audio channel to analyze (1-based)
            on_spectrum_callback: Callback for spectrum updates

        Returns:
            True if started successfully
        """
        if self.is_running:
            logger.warning("Spectrum analyzer already running")
            return False

        self.on_spectrum_updated = on_spectrum_callback
        self._init_essentia()
        self._target_channel = channel - 1  # 0-based index

        # Use unified AudioCapture if available
        if self._audio_capture is not None:
            try:
                self.sample_rate = self._audio_capture.sample_rate
                self._num_channels = channel
                self.freq_bins = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)
                self._audio_capture.subscribe('auto_eq', self._audio_capture_poll)
                self.is_running = True
                self._stop_event.clear()
                self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
                self._analysis_thread.start()
                logger.info(f"Spectrum analyzer started via AudioCapture: ch{channel}, {self.sample_rate}Hz")
                return True
            except Exception as e:
                logger.warning(f"AudioCapture integration failed, falling back to PyAudio: {e}")
                self._audio_capture = None

        # Fallback: direct PyAudio stream
        try:
            import pyaudio

            self.pa = pyaudio.PyAudio()

            # Get device info
            if self.device_index is not None:
                device_info = self.pa.get_device_info_by_index(int(self.device_index))
                max_channels = int(device_info.get('maxInputChannels', 2))
                self.sample_rate = int(device_info.get('defaultSampleRate', 48000))
                logger.info(f"Using device {self.device_index}: {device_info.get('name')}")
            else:
                device_info = self.pa.get_default_input_device_info()
                max_channels = int(device_info.get('maxInputChannels', 2))
                self.sample_rate = int(device_info.get('defaultSampleRate', 48000))
                logger.info(f"Using default device: {device_info.get('name')}")

            self._num_channels = min(channel, max_channels)

            # Update frequency bins for actual sample rate
            self.freq_bins = np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate)

            # Open audio stream
            self.stream = self.pa.open(
                format=pyaudio.paFloat32,
                channels=self._num_channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(self.device_index) if self.device_index is not None else None,
                frames_per_buffer=self.hop_size,
                stream_callback=self._audio_callback
            )

            self.stream.start_stream()
            self.is_running = True
            self._stop_event.clear()

            # Start analysis thread
            self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self._analysis_thread.start()

            logger.info(f"Spectrum analyzer started: {self._num_channels} channels, {self.sample_rate}Hz")
            return True

        except ImportError as e:
            logger.error(f"PyAudio not installed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error starting spectrum analyzer: {e}", exc_info=True)
            return False
    
    def stop(self):
        """Stop spectrum analysis."""
        self._stop_event.set()
        self.is_running = False

        if self._audio_capture is not None:
            try:
                self._audio_capture.unsubscribe('auto_eq')
            except Exception:
                pass

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None

        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None

        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=1.0)

        logger.info("Spectrum analyzer stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - receives audio data."""
        try:
            import pyaudio
            
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Extract target channel if multi-channel
            if self._num_channels > 1:
                audio_data = audio_data.reshape(-1, self._num_channels)
                channel_idx = min(self._target_channel, self._num_channels - 1)
                audio_data = audio_data[:, channel_idx]
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            return (None, pyaudio.paContinue)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            try:
                import pyaudio
                return (None, pyaudio.paContinue)
            except:
                return (None, 0)
    
    def _analysis_loop(self):
        """Background analysis thread."""
        while not self._stop_event.is_set() and self.is_running:
            try:
                if len(self.audio_buffer) >= self.frame_size:
                    # Get audio frame
                    audio_frame = np.array(list(self.audio_buffer)[-self.frame_size:])
                    
                    # Analyze
                    spectral_data = self.analyze(audio_frame)
                    
                    if spectral_data:
                        self.current_spectrum = spectral_data
                        
                        if self.on_spectrum_updated:
                            self.on_spectrum_updated(spectral_data)
                
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
            
            # ~30 fps update rate
            time.sleep(0.033)
    
    def analyze(self, audio_frame: np.ndarray) -> Optional[SpectralData]:
        """
        Analyze audio frame and return spectral data.
        
        Args:
            audio_frame: Audio samples (float32)
            
        Returns:
            SpectralData object with analysis results
        """
        if len(audio_frame) < self.frame_size:
            return None
        
        try:
            # Use Essentia if available, otherwise fallback to numpy
            if self._essentia_initialized:
                return self._analyze_essentia(audio_frame)
            else:
                return self._analyze_numpy(audio_frame)
                
        except Exception as e:
            logger.error(f"Error analyzing spectrum: {e}")
            return None
    
    def _analyze_essentia(self, audio_frame: np.ndarray) -> SpectralData:
        """Analyze using Essentia algorithms."""
        # Apply windowing
        windowed = self._windowing(audio_frame.astype(np.float32))
        
        # Compute spectrum
        spectrum = self._spectrum(windowed)
        
        # Compute spectral features
        peaks_freq, peaks_mag = self._spectral_peaks(spectrum)
        centroid = self._spectral_centroid(spectrum)
        flatness = self._spectral_flatness(spectrum)
        rolloff = self._rolloff(spectrum)
        
        # Convert centroid from normalized to Hz
        centroid_hz = centroid * (self.sample_rate / 2)
        
        # Find peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = self.freq_bins[peak_idx] if peak_idx < len(self.freq_bins) else 0
        
        # Calculate bandwidth (using standard deviation of spectrum)
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
        freqs = self.freq_bins[:len(spectrum)]
        mean_freq = np.sum(freqs * spectrum_norm)
        bandwidth = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * spectrum_norm))
        
        # Format peaks as list of tuples
        peaks = list(zip(peaks_freq.tolist(), peaks_mag.tolist()))
        
        return SpectralData(
            spectrum=spectrum,
            frequencies=self.freq_bins[:len(spectrum)],
            peak_freq=peak_freq,
            centroid=centroid_hz,
            rolloff=rolloff,
            flatness=float(flatness),
            bandwidth=bandwidth,
            peaks=peaks
        )
    
    def _analyze_numpy(self, audio_frame: np.ndarray) -> SpectralData:
        """Fallback analysis using numpy FFT."""
        # Apply Hann window
        window = np.hanning(len(audio_frame))
        windowed = audio_frame * window
        
        # Compute FFT
        fft = np.fft.rfft(windowed, n=self.frame_size)
        spectrum = np.abs(fft)
        
        # Normalize
        spectrum = spectrum / (self.frame_size / 2)
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Frequency bins
        freqs = self.freq_bins[:len(spectrum)]
        
        # Peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
        
        # Spectral centroid
        spectrum_sum = np.sum(spectrum)
        if spectrum_sum > 0:
            centroid = np.sum(freqs * spectrum) / spectrum_sum
        else:
            centroid = 0
        
        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
        arithmetic_mean = np.mean(spectrum)
        flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.where(cumsum >= 0.85 * cumsum[-1])[0]
        rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        # Bandwidth
        if spectrum_sum > 0:
            spectrum_norm = spectrum / spectrum_sum
            mean_freq = np.sum(freqs * spectrum_norm)
            bandwidth = np.sqrt(np.sum(((freqs - mean_freq) ** 2) * spectrum_norm))
        else:
            bandwidth = 0
        
        # Find peaks
        peak_indices, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.1, distance=10)
        peaks = [(freqs[i], spectrum[i]) for i in peak_indices[:20]]
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        return SpectralData(
            spectrum=spectrum,
            frequencies=freqs,
            peak_freq=peak_freq,
            centroid=centroid,
            rolloff=rolloff,
            flatness=flatness,
            bandwidth=bandwidth,
            peaks=peaks
        )
    
    def get_spectrum_for_visualization(self, num_bands: int = 32) -> List[float]:
        """
        Get spectrum data formatted for visualization (logarithmic frequency bands).
        
        Args:
            num_bands: Number of frequency bands for visualization
            
        Returns:
            List of dB values for each band
        """
        if self.current_spectrum is None:
            return [0.0] * num_bands
        
        spectrum = self.current_spectrum.spectrum
        freqs = self.current_spectrum.frequencies
        
        # Define logarithmic frequency bands (20Hz to 20kHz)
        band_edges = np.logspace(np.log10(20), np.log10(20000), num_bands + 1)
        
        band_values = []
        for i in range(num_bands):
            low_freq = band_edges[i]
            high_freq = band_edges[i + 1]
            
            # Find indices in this band
            mask = (freqs >= low_freq) & (freqs < high_freq)
            if np.any(mask):
                # Average magnitude in this band
                band_mag = np.mean(spectrum[mask])
                # Convert to dB
                band_db = 20 * np.log10(band_mag + 1e-10)
                # Clamp to reasonable range
                band_db = max(-60, min(0, band_db))
            else:
                band_db = -60
            
            band_values.append(band_db)
        
        return band_values


# ============================================================================
# Instrument Profiles
# ============================================================================

class InstrumentProfiles:
    """
    Collection of target EQ profiles for different instrument types.
    
    Each profile defines:
    - Target frequency curve
    - Problem frequencies to cut
    - Frequencies to boost
    - Low/high cut recommendations
    """
    
    # Standard frequency band centers for 1/3 octave analysis
    THIRD_OCTAVE_CENTERS = [
        20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
        630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
        10000, 12500, 16000, 20000
    ]
    
    @staticmethod
    def get_profile(instrument_type: str) -> Optional[InstrumentProfile]:
        """
        Get EQ profile for instrument type.
        
        Args:
            instrument_type: Instrument name (kick, snare, hihat, vocals, guitar, bass, keys, custom)
            
        Returns:
            InstrumentProfile or None if not found
        """
        profiles = {
            # Drums
            'kick': InstrumentProfiles._kick_profile(),
            'snare': InstrumentProfiles._snare_profile(),
            'tom': InstrumentProfiles._tom_profile(),
            'hihat': InstrumentProfiles._hihat_profile(),
            'cymbals': InstrumentProfiles._cymbals_profile(),
            'overheads': InstrumentProfiles._overheads_profile(),
            # Vocals
            'leadvocal': InstrumentProfiles._lead_vocal_profile(),
            'backvocal': InstrumentProfiles._back_vocal_profile(),
            'vocals': InstrumentProfiles._lead_vocal_profile(),
            # Vocal profiles by genre
            'vocal_rock': InstrumentProfiles._vocal_rock_profile(),
            'vocal_pop': InstrumentProfiles._vocal_pop_profile(),
            'vocal_jazz': InstrumentProfiles._vocal_jazz_profile(),
            'vocal_metal': InstrumentProfiles._vocal_metal_profile(),
            'vocal_rnb': InstrumentProfiles._vocal_rnb_profile(),
            'vocal_classical': InstrumentProfiles._vocal_classical_profile(),
            # Guitars
            'guitar': InstrumentProfiles._guitar_profile(),
            'acousticguitar': InstrumentProfiles._acoustic_guitar_profile(),
            # Guitar profiles by genre
            'guitar_rock': InstrumentProfiles._guitar_rock_profile(),
            'guitar_metal': InstrumentProfiles._guitar_metal_profile(),
            'guitar_jazz': InstrumentProfiles._guitar_jazz_profile(),
            'guitar_clean': InstrumentProfiles._guitar_clean_profile(),
            'guitar_lead': InstrumentProfiles._guitar_lead_profile(),
            # Bass
            'bass': InstrumentProfiles._bass_profile(),
            # Bass profiles by genre
            'bass_rock': InstrumentProfiles._bass_rock_profile(),
            'bass_metal': InstrumentProfiles._bass_metal_profile(),
            'bass_jazz': InstrumentProfiles._bass_jazz_profile(),
            'bass_funk': InstrumentProfiles._bass_funk_profile(),
            # Drums by genre
            'kick_rock': InstrumentProfiles._kick_rock_profile(),
            'kick_metal': InstrumentProfiles._kick_metal_profile(),
            'kick_jazz': InstrumentProfiles._kick_jazz_profile(),
            'kick_pop': InstrumentProfiles._kick_pop_profile(),
            'snare_rock': InstrumentProfiles._snare_rock_profile(),
            'snare_metal': InstrumentProfiles._snare_metal_profile(),
            'snare_jazz': InstrumentProfiles._snare_jazz_profile(),
            # Additional instruments
            'accordion': InstrumentProfiles._accordion_profile(),
            'playback': InstrumentProfiles._playback_profile(),
            'ftom': InstrumentProfiles._ftom_profile(),
            'ride': InstrumentProfiles._ride_profile(),
            # Orchestral instruments
            'violin': InstrumentProfiles._violin_profile(),
            'cello': InstrumentProfiles._cello_profile(),
            'trumpet': InstrumentProfiles._trumpet_profile(),
            'saxophone': InstrumentProfiles._saxophone_profile(),
            'flute': InstrumentProfiles._flute_profile(),
            # Keys
            'keys': InstrumentProfiles._keys_profile(),
            'piano': InstrumentProfiles._piano_profile(),
            'synth': InstrumentProfiles._synth_profile(),
            # Custom/Flat
            'custom': InstrumentProfiles._flat_profile(),
        }
        
        return profiles.get(instrument_type.lower())
    
    @staticmethod
    def get_all_profiles() -> List[str]:
        """Get list of all available profile names."""
        return [
            # Drums
            'kick', 'snare', 'tom', 'hihat', 'cymbals', 'overheads',
            'kick_rock', 'kick_metal', 'kick_jazz', 'kick_pop',
            'snare_rock', 'snare_metal', 'snare_jazz',
            'ftom', 'ride',
            # Vocals
            'leadvocal', 'backvocal', 'vocals',
            'vocal_rock', 'vocal_pop', 'vocal_jazz', 'vocal_metal', 'vocal_rnb', 'vocal_classical',
            # Guitars
            'guitar', 'acousticguitar',
            'guitar_rock', 'guitar_metal', 'guitar_jazz', 'guitar_clean', 'guitar_lead',
            # Bass
            'bass', 'bass_rock', 'bass_metal', 'bass_jazz', 'bass_funk',
            # Keys
            'keys', 'piano', 'synth',
            # Orchestral
            'violin', 'cello', 'trumpet', 'saxophone', 'flute',
            # Other
            'accordion', 'playback', 'custom'
        ]
    
    @staticmethod
    def _kick_profile() -> InstrumentProfile:
        """Kick drum EQ profile."""
        return InstrumentProfile(
            name='kick',
            description='Kick drum: sub punch, cut mud, click presence',
            target_curve=[
                (30, 3), (50, 4), (80, 2), (100, 0),
                (200, -3), (300, -4), (400, -3), (500, -2),
                (1000, 0), (2000, 0), (3000, 2), (4000, 3), (5000, 2),
                (8000, 0), (10000, -2), (16000, -6)
            ],
            cut_frequencies=[
                (250, -6, 2.0),   # Mud/boxy
                (400, -4, 1.5),   # Cardboard
            ],
            boost_frequencies=[
                (60, 4, 1.2),     # Sub punch
                (3500, 3, 2.0),   # Click/attack
            ],
            low_cut_freq=30,
            high_cut_freq=10000
        )
    
    @staticmethod
    def _snare_profile() -> InstrumentProfile:
        """Snare drum EQ profile."""
        return InstrumentProfile(
            name='snare',
            description='Snare drum: body, snap, air',
            target_curve=[
                (80, -2), (100, 0), (150, 2), (200, 3), (250, 2),
                (400, -2), (500, -1), (800, 0),
                (2000, 2), (3000, 3), (4000, 3), (5000, 2),
                (8000, 1), (10000, 2), (12000, 1)
            ],
            cut_frequencies=[
                (400, -4, 2.0),   # Boxy
                (800, -2, 1.5),   # Hollow
            ],
            boost_frequencies=[
                (180, 3, 1.5),    # Body/fatness
                (3000, 3, 2.0),   # Snap/crack
                (10000, 2, 0.7),  # Air/sizzle
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _tom_profile() -> InstrumentProfile:
        """Tom drum EQ profile."""
        return InstrumentProfile(
            name='tom',
            description='Tom drum: fundamental, attack',
            target_curve=[
                (60, 2), (80, 3), (100, 2), (150, 1),
                (250, -2), (400, -3), (500, -2),
                (2000, 1), (3000, 2), (4000, 2), (5000, 1),
                (8000, 0), (10000, -2)
            ],
            cut_frequencies=[
                (300, -4, 2.0),   # Boxy
                (600, -3, 1.5),   # Ring
            ],
            boost_frequencies=[
                (80, 3, 1.2),     # Fundamental
                (3500, 2, 2.0),   # Attack
            ],
            low_cut_freq=50,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _hihat_profile() -> InstrumentProfile:
        """Hi-hat EQ profile."""
        return InstrumentProfile(
            name='hihat',
            description='Hi-hat: presence, sizzle, cut low',
            target_curve=[
                (100, -12), (200, -10), (300, -8), (500, -6),
                (1000, -4), (2000, -2), (4000, 0),
                (6000, 2), (8000, 3), (10000, 3), (12000, 2), (16000, 1)
            ],
            cut_frequencies=[
                (400, -6, 1.0),   # Mud
            ],
            boost_frequencies=[
                (8000, 3, 1.5),   # Sizzle
                (12000, 2, 0.7),  # Air
            ],
            low_cut_freq=300,
            high_cut_freq=None
        )
    
    @staticmethod
    def _cymbals_profile() -> InstrumentProfile:
        """Cymbals EQ profile."""
        return InstrumentProfile(
            name='cymbals',
            description='Cymbals: shimmer, air',
            target_curve=[
                (100, -10), (200, -8), (400, -6), (800, -4),
                (1500, -2), (3000, 0), (5000, 1),
                (8000, 2), (10000, 2), (12000, 2), (16000, 1)
            ],
            cut_frequencies=[
                (500, -4, 1.0),   # Low mud
            ],
            boost_frequencies=[
                (10000, 2, 1.0),  # Shimmer
            ],
            low_cut_freq=400,
            high_cut_freq=None
        )
    
    @staticmethod
    def _overheads_profile() -> InstrumentProfile:
        """Drum overheads EQ profile."""
        return InstrumentProfile(
            name='overheads',
            description='Overheads: balanced kit image, air',
            target_curve=[
                (60, -4), (100, -2), (200, 0), (400, -1),
                (800, 0), (1500, 0), (3000, 1),
                (6000, 1), (8000, 2), (10000, 2), (12000, 1), (16000, 0)
            ],
            cut_frequencies=[
                (300, -3, 1.5),   # Boxy room
            ],
            boost_frequencies=[
                (8000, 2, 0.7),   # Air
            ],
            low_cut_freq=80,
            high_cut_freq=None
        )
    
    @staticmethod
    def _lead_vocal_profile() -> InstrumentProfile:
        """Lead vocal EQ profile."""
        return InstrumentProfile(
            name='leadvocal',
            description='Lead vocal: clarity, presence, air',
            target_curve=[
                (80, -6), (100, -4), (150, -2), (200, 0),
                (300, -1), (500, 0), (800, 0),
                (1500, 0), (2500, 2), (3500, 3), (5000, 2),
                (8000, 1), (10000, 2), (12000, 1)
            ],
            cut_frequencies=[
                (250, -3, 2.0),   # Mud
                (800, -2, 2.5),   # Honky/nasal
                (4000, -2, 3.0),  # Harsh sibilance
            ],
            boost_frequencies=[
                (3000, 3, 2.0),   # Presence
                (10000, 2, 0.7),  # Air
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _back_vocal_profile() -> InstrumentProfile:
        """Background vocal EQ profile."""
        return InstrumentProfile(
            name='backvocal',
            description='Background vocal: sit behind lead, blend',
            target_curve=[
                (80, -8), (100, -6), (150, -4), (200, -2),
                (300, -2), (500, 0), (800, 0),
                (1500, 0), (2500, 1), (3500, 1), (5000, 0),
                (8000, 0), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (200, -4, 1.5),   # Mud
                (3000, -2, 2.0),  # Don't compete with lead
            ],
            boost_frequencies=[
                (10000, 1, 0.7),  # Air/blend
            ],
            low_cut_freq=120,
            high_cut_freq=14000
        )
    
    @staticmethod
    def _guitar_profile() -> InstrumentProfile:
        """Electric guitar EQ profile."""
        return InstrumentProfile(
            name='guitar',
            description='Electric guitar: cut mud, presence',
            target_curve=[
                (80, -6), (100, -4), (150, -2), (200, -1),
                (300, -2), (400, -3), (500, -2), (800, 0),
                (1500, 0), (2500, 2), (3500, 2), (5000, 1),
                (8000, 0), (10000, -2)
            ],
            cut_frequencies=[
                (300, -4, 2.0),   # Mud
                (800, -2, 2.0),   # Honky
            ],
            boost_frequencies=[
                (3000, 2, 2.0),   # Presence/bite
            ],
            low_cut_freq=100,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _acoustic_guitar_profile() -> InstrumentProfile:
        """Acoustic guitar EQ profile."""
        return InstrumentProfile(
            name='acousticguitar',
            description='Acoustic guitar: body, sparkle',
            target_curve=[
                (80, -4), (100, -2), (150, 0), (200, 1),
                (300, -1), (400, -2), (500, -1), (800, 0),
                (1500, 0), (2500, 1), (4000, 2), (6000, 2),
                (8000, 1), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (300, -3, 2.0),   # Mud/boom
                (2000, -2, 3.0),  # Harsh pick
            ],
            boost_frequencies=[
                (180, 2, 1.5),    # Body
                (5000, 2, 1.5),   # Sparkle
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _bass_profile() -> InstrumentProfile:
        """Bass guitar EQ profile."""
        return InstrumentProfile(
            name='bass',
            description='Bass: fundamental, harmonics, cut mud',
            target_curve=[
                (40, 2), (60, 3), (80, 3), (100, 2),
                (150, 0), (200, -2), (300, -3), (400, -2),
                (600, 0), (800, 1), (1000, 2), (1500, 1),
                (3000, 0), (5000, -2), (8000, -6)
            ],
            cut_frequencies=[
                (250, -4, 2.0),   # Mud
                (400, -3, 1.5),   # Boxy
            ],
            boost_frequencies=[
                (80, 3, 1.2),     # Fundamental
                (900, 2, 2.0),    # Growl/harmonics
            ],
            low_cut_freq=30,
            high_cut_freq=8000
        )
    
    @staticmethod
    def _keys_profile() -> InstrumentProfile:
        """Keyboards/synth EQ profile."""
        return InstrumentProfile(
            name='keys',
            description='Keys: balanced, slight mud cut',
            target_curve=[
                (60, 0), (100, 0), (200, 0),
                (300, -2), (400, -2), (500, -1),
                (1000, 0), (2000, 0), (4000, 0),
                (8000, 0), (12000, 0), (16000, -2)
            ],
            cut_frequencies=[
                (350, -3, 1.5),   # Mud
            ],
            boost_frequencies=[],
            low_cut_freq=60,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _piano_profile() -> InstrumentProfile:
        """Piano EQ profile."""
        return InstrumentProfile(
            name='piano',
            description='Piano: full range, presence',
            target_curve=[
                (40, 1), (60, 1), (80, 1), (100, 0),
                (200, 0), (300, -1), (400, -1), (500, 0),
                (1000, 0), (2000, 1), (3000, 1), (4000, 1),
                (6000, 0), (8000, 0), (10000, -1), (16000, -2)
            ],
            cut_frequencies=[
                (300, -2, 1.5),   # Mud
            ],
            boost_frequencies=[
                (3000, 2, 2.0),   # Presence
            ],
            low_cut_freq=40,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _synth_profile() -> InstrumentProfile:
        """Synth EQ profile."""
        return InstrumentProfile(
            name='synth',
            description='Synth: full range, context dependent',
            target_curve=[
                (40, 0), (80, 0), (150, 0), (300, 0),
                (600, 0), (1200, 0), (2500, 0), (5000, 0),
                (10000, 0), (16000, 0)
            ],
            cut_frequencies=[],
            boost_frequencies=[],
            low_cut_freq=None,
            high_cut_freq=None
        )
    
    @staticmethod
    def _flat_profile() -> InstrumentProfile:
        """Flat/custom EQ profile."""
        return InstrumentProfile(
            name='custom',
            description='Flat response - no processing',
            target_curve=[
                (20, 0), (100, 0), (500, 0), (1000, 0),
                (5000, 0), (10000, 0), (20000, 0)
            ],
            cut_frequencies=[],
            boost_frequencies=[],
            low_cut_freq=None,
            high_cut_freq=None
        )
    
    # ========== Vocal Profiles by Genre ==========
    
    @staticmethod
    def _vocal_rock_profile() -> InstrumentProfile:
        """Rock vocal EQ profile."""
        return InstrumentProfile(
            name='vocal_rock',
            description='Rock vocal: aggressive, presence, cut lows',
            target_curve=[
                (80, -8), (100, -6), (150, -4), (200, -2),
                (300, -1), (500, 0), (800, 0),
                (1500, 1), (2500, 2), (3500, 3), (5000, 3), (6000, 2),
                (8000, 1), (10000, 2), (12000, 1)
            ],
            cut_frequencies=[
                (120, -6, 1.5),   # Low cut
                (250, -3, 2.0),   # Mud
                (4000, -2, 3.0),  # Harsh sibilance
            ],
            boost_frequencies=[
                (3500, 3, 2.0),   # Presence
                (10000, 2, 0.7),  # Air
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _vocal_pop_profile() -> InstrumentProfile:
        """Pop vocal EQ profile."""
        return InstrumentProfile(
            name='vocal_pop',
            description='Pop vocal: bright, airy, cut lows',
            target_curve=[
                (80, -10), (100, -8), (150, -6), (200, -4),
                (300, -2), (500, 0), (800, 0),
                (1500, 1), (2500, 2), (4000, 2), (6000, 2),
                (8000, 2), (10000, 3), (12000, 2), (16000, 1)
            ],
            cut_frequencies=[
                (100, -6, 1.5),   # Low cut
                (300, -2, 2.0),   # Mud
            ],
            boost_frequencies=[
                (5000, 2, 1.5),   # Brightness
                (10000, 3, 0.7),  # Air
            ],
            low_cut_freq=100,
            high_cut_freq=18000
        )
    
    @staticmethod
    def _vocal_jazz_profile() -> InstrumentProfile:
        """Jazz vocal EQ profile."""
        return InstrumentProfile(
            name='vocal_jazz',
            description='Jazz vocal: warm, natural, balanced',
            target_curve=[
                (80, -4), (100, -2), (150, 0), (200, 1),
                (300, 1), (500, 1), (800, 0),
                (1500, 0), (2500, 1), (3500, 1), (5000, 1),
                (8000, 1), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (200, -2, 1.5),   # Slight mud
            ],
            boost_frequencies=[
                (300, 1, 1.5),    # Warmth
                (5000, 1, 1.5),   # Clarity
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _vocal_metal_profile() -> InstrumentProfile:
        """Metal vocal EQ profile."""
        return InstrumentProfile(
            name='vocal_metal',
            description='Metal vocal: aggressive, mids, cut lows',
            target_curve=[
                (80, -10), (100, -8), (120, -6), (150, -4),
                (300, -1), (500, 0), (1000, 1),
                (2000, 2), (3000, 3), (4000, 3), (5000, 2),
                (8000, 1), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (120, -6, 1.5),   # Low cut
                (400, -2, 2.0),   # Mud
                (6000, -2, 2.5),  # Harsh
            ],
            boost_frequencies=[
                (2500, 3, 2.0),   # Aggressive mids
                (4000, 2, 2.0),   # Presence
            ],
            low_cut_freq=120,
            high_cut_freq=14000
        )
    
    @staticmethod
    def _vocal_rnb_profile() -> InstrumentProfile:
        """R&B vocal EQ profile."""
        return InstrumentProfile(
            name='vocal_rnb',
            description='R&B vocal: deep bass, airy top',
            target_curve=[
                (80, -2), (100, 0), (150, 1), (200, 2),
                (300, 1), (500, 0), (800, 0),
                (1500, 1), (2500, 2), (4000, 2), (6000, 2),
                (8000, 2), (10000, 3), (12000, 2), (16000, 1)
            ],
            cut_frequencies=[
                (250, -2, 1.5),   # Slight mud
            ],
            boost_frequencies=[
                (150, 2, 1.2),    # Deep bass
                (10000, 3, 0.7),  # Air
                (12000, 2, 0.5),  # Shimmer
            ],
            low_cut_freq=80,
            high_cut_freq=18000
        )
    
    @staticmethod
    def _vocal_classical_profile() -> InstrumentProfile:
        """Classical vocal EQ profile."""
        return InstrumentProfile(
            name='vocal_classical',
            description='Classical vocal: natural, wide range, minimal processing',
            target_curve=[
                (60, 0), (100, 0), (200, 0), (500, 0),
                (1000, 0), (2000, 0), (4000, 0), (8000, 0),
                (10000, 0), (12000, 0), (16000, 0)
            ],
            cut_frequencies=[
                (150, -1, 1.0),   # Very slight low cut
            ],
            boost_frequencies=[],
            low_cut_freq=60,
            high_cut_freq=None
        )
    
    # ========== Guitar Profiles by Genre ==========
    
    @staticmethod
    def _guitar_rock_profile() -> InstrumentProfile:
        """Rock guitar EQ profile."""
        return InstrumentProfile(
            name='guitar_rock',
            description='Rock guitar: aggressive mids, cut mud',
            target_curve=[
                (80, -8), (100, -6), (150, -4), (200, -2),
                (300, -2), (500, 0), (800, -1),
                (1500, 2), (2500, 3), (3500, 3), (5000, 2),
                (8000, 1), (10000, 0), (12000, -2)
            ],
            cut_frequencies=[
                (100, -6, 1.5),   # Low cut
                (300, -4, 2.0),   # Mud
                (800, -2, 2.0),   # Honky
            ],
            boost_frequencies=[
                (2000, 3, 2.0),   # Mids/presence
                (5000, 2, 2.0),   # Bite
            ],
            low_cut_freq=100,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _guitar_metal_profile() -> InstrumentProfile:
        """Metal guitar EQ profile."""
        return InstrumentProfile(
            name='guitar_metal',
            description='Metal guitar: aggressive mids, cut lows',
            target_curve=[
                (80, -10), (100, -8), (120, -6), (200, -4),
                (300, -2), (500, 0), (1000, 1),
                (2000, 2), (3000, 3), (4000, 3), (5000, 2),
                (8000, 1), (10000, 0), (12000, -2)
            ],
            cut_frequencies=[
                (120, -6, 1.5),   # Low cut
                (400, -3, 2.0),   # Mud
                (600, -2, 1.5),   # Honky
            ],
            boost_frequencies=[
                (2500, 3, 2.0),   # Aggressive mids
                (4000, 3, 2.0),   # Presence
            ],
            low_cut_freq=120,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _guitar_jazz_profile() -> InstrumentProfile:
        """Jazz guitar EQ profile."""
        return InstrumentProfile(
            name='guitar_jazz',
            description='Jazz guitar: warm, natural, balanced',
            target_curve=[
                (80, -4), (100, -2), (150, 0), (200, 1),
                (300, 0), (500, 0), (800, 0),
                (1500, 0), (2500, 1), (4000, 1), (6000, 1),
                (8000, 0), (10000, 0), (12000, -1)
            ],
            cut_frequencies=[
                (200, -2, 1.5),   # Slight mud
            ],
            boost_frequencies=[
                (400, 1, 1.5),    # Warmth
                (5000, 1, 1.5),   # Clarity
            ],
            low_cut_freq=80,
            high_cut_freq=14000
        )
    
    @staticmethod
    def _guitar_clean_profile() -> InstrumentProfile:
        """Clean guitar EQ profile."""
        return InstrumentProfile(
            name='guitar_clean',
            description='Clean guitar: balanced, clear',
            target_curve=[
                (80, -4), (100, -2), (150, 0), (200, 0),
                (300, -1), (500, 0), (800, 0),
                (1500, 0), (2500, 1), (4000, 1), (6000, 1),
                (8000, 1), (10000, 0), (12000, 0)
            ],
            cut_frequencies=[
                (250, -2, 1.5),   # Mud
            ],
            boost_frequencies=[
                (5000, 1, 1.5),   # Clarity
            ],
            low_cut_freq=100,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _guitar_lead_profile() -> InstrumentProfile:
        """Lead guitar EQ profile."""
        return InstrumentProfile(
            name='guitar_lead',
            description='Lead guitar: presence, bright top',
            target_curve=[
                (80, -6), (100, -4), (150, -2), (200, -1),
                (300, -1), (500, 0), (1000, 1),
                (2000, 2), (3000, 3), (4000, 3), (5000, 3),
                (8000, 2), (10000, 2), (12000, 1)
            ],
            cut_frequencies=[
                (150, -4, 1.5),   # Low cut
                (400, -2, 2.0),   # Mud
            ],
            boost_frequencies=[
                (3500, 3, 2.0),   # Presence
                (5000, 2, 1.5),   # Brightness
            ],
            low_cut_freq=100,
            high_cut_freq=16000
        )
    
    # ========== Bass Profiles by Genre ==========
    
    @staticmethod
    def _bass_rock_profile() -> InstrumentProfile:
        """Rock bass EQ profile."""
        return InstrumentProfile(
            name='bass_rock',
            description='Rock bass: aggressive, mids, foundation',
            target_curve=[
                (40, 2), (60, 3), (80, 3), (100, 2),
                (200, 0), (400, 0), (600, 1),
                (800, 2), (1200, 2), (1500, 1),
                (3000, 0), (5000, -2), (8000, -4)
            ],
            cut_frequencies=[
                (250, -3, 2.0),   # Mud
                (500, -2, 1.5),   # Boxy
            ],
            boost_frequencies=[
                (70, 3, 1.2),     # Foundation
                (900, 2, 2.0),   # Growl/mids
            ],
            low_cut_freq=40,
            high_cut_freq=8000
        )
    
    @staticmethod
    def _bass_metal_profile() -> InstrumentProfile:
        """Metal bass EQ profile."""
        return InstrumentProfile(
            name='bass_metal',
            description='Metal bass: heavy lows, cut mids',
            target_curve=[
                (30, 2), (50, 3), (70, 4), (100, 3),
                (200, 0), (400, -3), (600, -4),
                (800, -2), (1200, 0), (2000, 1),
                (4000, 0), (6000, -2), (8000, -4)
            ],
            cut_frequencies=[
                (400, -5, 2.0),   # Mud/mids
                (600, -4, 1.5),   # Boxy
            ],
            boost_frequencies=[
                (60, 4, 1.0),     # Heavy lows
                (2000, 1, 2.0),   # Definition
            ],
            low_cut_freq=30,
            high_cut_freq=8000
        )
    
    @staticmethod
    def _bass_jazz_profile() -> InstrumentProfile:
        """Jazz bass EQ profile."""
        return InstrumentProfile(
            name='bass_jazz',
            description='Jazz bass: warm, natural, wide range',
            target_curve=[
                (40, 1), (60, 2), (80, 2), (100, 1),
                (200, 0), (400, 0), (600, 0),
                (800, 0), (1200, 1), (2000, 1),
                (4000, 1), (6000, 0), (8000, 0)
            ],
            cut_frequencies=[
                (300, -2, 1.5),   # Slight mud
            ],
            boost_frequencies=[
                (80, 2, 1.2),     # Warmth
                (1500, 1, 2.0),  # Clarity
            ],
            low_cut_freq=40,
            high_cut_freq=10000
        )
    
    @staticmethod
    def _bass_funk_profile() -> InstrumentProfile:
        """Funk bass EQ profile."""
        return InstrumentProfile(
            name='bass_funk',
            description='Funk bass: bright, harmonics, clear low',
            target_curve=[
                (40, 1), (60, 2), (80, 2), (100, 1),
                (200, 0), (400, 0), (600, 1),
                (800, 2), (1200, 3), (2000, 3), (3000, 2),
                (5000, 1), (7000, 0), (10000, -2)
            ],
            cut_frequencies=[
                (250, -2, 1.5),   # Mud
            ],
            boost_frequencies=[
                (80, 2, 1.2),     # Clear low
                (1500, 3, 2.0),   # Bright harmonics
            ],
            low_cut_freq=40,
            high_cut_freq=10000
        )
    
    # ========== Drum Profiles by Genre ==========
    
    @staticmethod
    def _kick_rock_profile() -> InstrumentProfile:
        """Rock kick drum EQ profile."""
        return InstrumentProfile(
            name='kick_rock',
            description='Rock kick: aggressive, click, foundation',
            target_curve=[
                (30, 2), (50, 3), (60, 3), (80, 2),
                (200, -3), (300, -4), (400, -3), (500, -2),
                (1000, 0), (2000, 0), (3000, 2), (4000, 3), (5000, 2),
                (8000, 0), (10000, -2), (16000, -6)
            ],
            cut_frequencies=[
                (250, -6, 2.0),   # Mud/boxy
                (400, -4, 1.5),   # Cardboard
            ],
            boost_frequencies=[
                (60, 3, 1.2),     # Foundation
                (3500, 3, 2.0),   # Click/attack
            ],
            low_cut_freq=30,
            high_cut_freq=10000
        )
    
    @staticmethod
    def _kick_metal_profile() -> InstrumentProfile:
        """Metal kick drum EQ profile."""
        return InstrumentProfile(
            name='kick_metal',
            description='Metal kick: heavy click, cut lows',
            target_curve=[
                (30, 1), (50, 2), (60, 2), (80, 1),
                (200, -4), (300, -5), (400, -4), (500, -3),
                (1000, -1), (2000, 0), (4000, 3), (5000, 4), (6000, 3),
                (8000, 1), (10000, 0), (16000, -4)
            ],
            cut_frequencies=[
                (250, -6, 2.0),   # Mud
                (50, -3, 1.0),    # Cut excessive sub
            ],
            boost_frequencies=[
                (4500, 4, 2.0),   # Heavy click
                (5500, 3, 1.5),   # Attack
            ],
            low_cut_freq=50,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _kick_jazz_profile() -> InstrumentProfile:
        """Jazz kick drum EQ profile."""
        return InstrumentProfile(
            name='kick_jazz',
            description='Jazz kick: natural, warm, less processing',
            target_curve=[
                (40, 1), (60, 2), (80, 2), (100, 1),
                (200, -2), (300, -2), (400, -1), (500, 0),
                (1000, 0), (2000, 1), (3000, 1), (4000, 1),
                (8000, 0), (10000, -1), (16000, -3)
            ],
            cut_frequencies=[
                (300, -3, 2.0),   # Boxy
            ],
            boost_frequencies=[
                (70, 2, 1.2),     # Natural low
            ],
            low_cut_freq=40,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _kick_pop_profile() -> InstrumentProfile:
        """Pop kick drum EQ profile."""
        return InstrumentProfile(
            name='kick_pop',
            description='Pop kick: balanced, bright click',
            target_curve=[
                (30, 2), (50, 3), (60, 3), (80, 2),
                (200, -2), (300, -3), (400, -2), (500, -1),
                (1000, 0), (2000, 1), (3000, 2), (4000, 3), (5000, 2),
                (8000, 1), (10000, 0), (16000, -2)
            ],
            cut_frequencies=[
                (250, -5, 2.0),   # Mud
                (400, -3, 1.5),   # Cardboard
            ],
            boost_frequencies=[
                (60, 3, 1.2),     # Foundation
                (4000, 3, 2.0),   # Bright click
            ],
            low_cut_freq=30,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _snare_rock_profile() -> InstrumentProfile:
        """Rock snare drum EQ profile."""
        return InstrumentProfile(
            name='snare_rock',
            description='Rock snare: aggressive, snap, body',
            target_curve=[
                (80, -2), (100, 0), (150, 2), (200, 3), (250, 2),
                (400, -3), (500, -2), (800, -1),
                (2000, 2), (3000, 3), (4000, 3), (5000, 2),
                (8000, 1), (10000, 2), (12000, 1)
            ],
            cut_frequencies=[
                (400, -5, 2.0),   # Boxy
                (800, -3, 1.5),   # Hollow
            ],
            boost_frequencies=[
                (180, 3, 1.5),    # Body
                (3000, 3, 2.0),   # Aggressive snap
                (10000, 2, 0.7),  # Air
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _snare_metal_profile() -> InstrumentProfile:
        """Metal snare drum EQ profile."""
        return InstrumentProfile(
            name='snare_metal',
            description='Metal snare: very aggressive, heavy snap',
            target_curve=[
                (80, -4), (100, -2), (150, 0), (200, 1),
                (400, -4), (500, -3), (800, -2),
                (2000, 1), (3000, 2), (4000, 4), (5000, 4), (6000, 3),
                (8000, 2), (10000, 2), (12000, 1)
            ],
            cut_frequencies=[
                (400, -6, 2.0),   # Boxy
                (200, -3, 1.5),   # Cut lows
            ],
            boost_frequencies=[
                (4500, 4, 2.0),   # Heavy snap
                (5500, 3, 1.5),   # Crack
            ],
            low_cut_freq=120,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _snare_jazz_profile() -> InstrumentProfile:
        """Jazz snare drum EQ profile."""
        return InstrumentProfile(
            name='snare_jazz',
            description='Jazz snare: natural, soft, less processing',
            target_curve=[
                (80, -1), (100, 0), (150, 1), (200, 2), (250, 1),
                (400, -2), (500, -1), (800, 0),
                (2000, 1), (3000, 1), (4000, 1), (5000, 1),
                (8000, 0), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (400, -3, 2.0),   # Boxy
            ],
            boost_frequencies=[
                (180, 2, 1.5),    # Natural body
            ],
            low_cut_freq=80,
            high_cut_freq=16000
        )
    
    # ========== Additional Instruments ==========
    
    @staticmethod
    def _accordion_profile() -> InstrumentProfile:
        """Accordion EQ profile."""
        return InstrumentProfile(
            name='accordion',
            description='Accordion: warm mids, bright upper range',
            target_curve=[
                (100, -6), (150, -4), (200, -2), (300, 1),
                (500, 2), (800, 2), (1000, 1), (1500, 1),
                (3000, 2), (5000, 3), (6000, 2), (8000, 1),
                (10000, 0), (12000, -2)
            ],
            cut_frequencies=[
                (150, -6, 1.5),   # Low cut
                (2000, -2, 2.5),  # Harsh reeds
            ],
            boost_frequencies=[
                (600, 2, 1.5),    # Warm mids
                (4500, 3, 2.0),   # Brightness
            ],
            low_cut_freq=150,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _playback_profile() -> InstrumentProfile:
        """Playback EQ profile."""
        return InstrumentProfile(
            name='playback',
            description='Playback: balanced spectrum, slight mud cut',
            target_curve=[
                (60, 0), (100, 0), (200, 0), (300, -1),
                (400, -2), (500, -1), (800, 0),
                (1500, 0), (3000, 0), (5000, 0),
                (8000, 0), (10000, 0), (16000, 0)
            ],
            cut_frequencies=[
                (300, -3, 1.5),   # Mud
            ],
            boost_frequencies=[],
            low_cut_freq=60,
            high_cut_freq=16000
        )
    
    @staticmethod
    def _ftom_profile() -> InstrumentProfile:
        """Floor tom EQ profile."""
        return InstrumentProfile(
            name='ftom',
            description='Floor tom: deep low end, attack, cut boxy',
            target_curve=[
                (50, 2), (60, 3), (80, 3), (100, 2),
                (150, 0), (250, -2), (300, -4), (400, -3), (500, -2),
                (2000, 1), (3000, 2), (4000, 2), (5000, 1),
                (8000, 0), (10000, -2)
            ],
            cut_frequencies=[
                (350, -5, 2.0),   # Boxy
                (600, -3, 1.5),   # Ring
            ],
            boost_frequencies=[
                (70, 3, 1.2),     # Deep fundamental
                (3500, 2, 2.0),   # Attack
            ],
            low_cut_freq=40,
            high_cut_freq=12000
        )
    
    @staticmethod
    def _ride_profile() -> InstrumentProfile:
        """Ride cymbal EQ profile."""
        return InstrumentProfile(
            name='ride',
            description='Ride cymbal: bright shimmer, cut low',
            target_curve=[
                (100, -12), (200, -10), (400, -8), (600, -6),
                (1000, -4), (2000, -2), (4000, 0),
                (6000, 1), (8000, 2), (10000, 3), (12000, 3), (15000, 2), (16000, 1)
            ],
            cut_frequencies=[
                (500, -6, 1.0),   # Low mud
            ],
            boost_frequencies=[
                (10000, 3, 1.0),  # Shimmer
                (12000, 2, 0.7),  # Brightness
            ],
            low_cut_freq=500,
            high_cut_freq=None
        )
    
    # ========== Orchestral Instruments ==========
    
    @staticmethod
    def _violin_profile() -> InstrumentProfile:
        """Violin EQ profile."""
        return InstrumentProfile(
            name='violin',
            description='Violin: bright upper range, warm body',
            target_curve=[
                (100, -4), (150, -2), (200, 0), (300, 1),
                (400, 2), (500, 1), (800, 0),
                (1500, 1), (3000, 2), (5000, 3), (8000, 3),
                (10000, 2), (12000, 1), (16000, 0)
            ],
            cut_frequencies=[
                (150, -4, 1.5),   # Low cut
                (2000, -2, 3.0),  # Harsh bow
            ],
            boost_frequencies=[
                (400, 2, 1.5),    # Body/warmth
                (6000, 3, 1.5),   # Brightness
            ],
            low_cut_freq=150,
            high_cut_freq=18000
        )
    
    @staticmethod
    def _cello_profile() -> InstrumentProfile:
        """Cello EQ profile."""
        return InstrumentProfile(
            name='cello',
            description='Cello: deep low end, warm upper range',
            target_curve=[
                (60, 1), (80, 2), (100, 2), (150, 1),
                (200, 0), (300, 0), (500, 0), (800, 0),
                (1500, 1), (2000, 2), (3000, 2), (5000, 2),
                (8000, 1), (10000, 0), (12000, -1)
            ],
            cut_frequencies=[
                (250, -2, 1.5),   # Mud
            ],
            boost_frequencies=[
                (90, 2, 1.2),     # Deep low
                (2500, 2, 1.5),   # Warmth
            ],
            low_cut_freq=60,
            high_cut_freq=14000
        )
    
    @staticmethod
    def _trumpet_profile() -> InstrumentProfile:
        """Trumpet EQ profile."""
        return InstrumentProfile(
            name='trumpet',
            description='Trumpet: bright upper range, cut low',
            target_curve=[
                (100, -8), (150, -6), (200, -4), (300, -2),
                (500, 0), (800, 1), (1500, 2),
                (2000, 3), (3000, 3), (5000, 3), (6000, 2),
                (8000, 1), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (200, -6, 1.5),   # Low cut
                (1000, -2, 2.0),  # Honky
            ],
            boost_frequencies=[
                (2500, 3, 2.0),   # Brightness
                (5000, 2, 1.5),   # Presence
            ],
            low_cut_freq=200,
            high_cut_freq=14000
        )
    
    @staticmethod
    def _saxophone_profile() -> InstrumentProfile:
        """Saxophone EQ profile."""
        return InstrumentProfile(
            name='saxophone',
            description='Saxophone: warm mids, bright upper range',
            target_curve=[
                (80, -4), (100, -2), (150, 0), (200, 1),
                (300, 1), (500, 2), (800, 2), (1000, 1),
                (1500, 1), (2000, 2), (3000, 2), (5000, 3), (6000, 2),
                (8000, 1), (10000, 1), (12000, 0)
            ],
            cut_frequencies=[
                (150, -3, 1.5),   # Low cut
            ],
            boost_frequencies=[
                (600, 2, 1.5),    # Warm mids
                (5000, 3, 1.5),   # Brightness
            ],
            low_cut_freq=150,
            high_cut_freq=14000
        )
    
    @staticmethod
    def _flute_profile() -> InstrumentProfile:
        """Flute EQ profile."""
        return InstrumentProfile(
            name='flute',
            description='Flute: airy upper range, cut low',
            target_curve=[
                (100, -8), (150, -6), (200, -4), (300, -2),
                (500, 0), (1000, 1), (2000, 2),
                (3000, 2), (4000, 3), (6000, 3), (8000, 3),
                (10000, 2), (12000, 1), (16000, 0)
            ],
            cut_frequencies=[
                (300, -6, 1.5),   # Low cut
                (1500, -2, 2.0),  # Honky
            ],
            boost_frequencies=[
                (5000, 3, 1.5),   # Airy brightness
                (8000, 2, 1.0),   # Air
            ],
            low_cut_freq=300,
            high_cut_freq=18000
        )


# ============================================================================
# EQ Corrector (SciPy)
# ============================================================================

class EQCorrector:
    """
    Calculates EQ corrections based on spectrum analysis and target profile.
    
    Uses SciPy for:
    - Spectrum smoothing (Savitzky-Golay filter)
    - Peak/valley detection
    - Optimal Q factor calculation
    """
    
    def __init__(self, 
                 max_gain: float = 12.0,
                 min_gain: float = -12.0,
                 smoothing_window: int = 11,
                 peak_prominence: float = 3.0):
        """
        Initialize EQ corrector.
        
        Args:
            max_gain: Maximum boost in dB
            min_gain: Maximum cut in dB (negative)
            smoothing_window: Window size for Savitzky-Golay filter
            peak_prominence: Minimum prominence for peak detection
        """
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.smoothing_window = smoothing_window
        self.peak_prominence = peak_prominence
    
    def calculate_correction(self, 
                            spectral_data: SpectralData, 
                            profile: InstrumentProfile,
                            sensitivity: float = 1.0) -> List[EQBand]:
        """
        Calculate EQ correction bands based on current spectrum and target profile.
        
        Args:
            spectral_data: Current spectral analysis data
            profile: Target instrument profile
            sensitivity: Correction sensitivity (0.0 to 2.0, default 1.0)
            
        Returns:
            List of EQBand corrections to apply
        """
        corrections = []
        
        # Get spectrum in dB
        spectrum = spectral_data.spectrum
        freqs = spectral_data.frequencies
        
        # Convert to dB
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        
        # Smooth spectrum
        if len(spectrum_db) > self.smoothing_window:
            spectrum_smooth = savgol_filter(spectrum_db, self.smoothing_window, 3)
        else:
            spectrum_smooth = spectrum_db
        
        # Create target curve interpolation
        target_freqs = [p[0] for p in profile.target_curve]
        target_gains = [p[1] for p in profile.target_curve]
        target_interp = interp1d(target_freqs, target_gains, 
                                  kind='linear', 
                                  bounds_error=False, 
                                  fill_value=(target_gains[0], target_gains[-1]))
        
        # Get target values at spectrum frequencies
        target_curve = target_interp(freqs)
        
        # Calculate deviation from target
        # Normalize spectrum to 0 dB peak
        spectrum_normalized = spectrum_smooth - np.max(spectrum_smooth)
        deviation = target_curve - spectrum_normalized
        
        # Find significant deviations (peaks and valleys)
        # Positive deviation = need boost, negative = need cut
        
        # Apply profile's cut frequencies
        for freq, max_cut, q in profile.cut_frequencies:
            # Find closest frequency in spectrum
            freq_idx = np.argmin(np.abs(freqs - freq))
            if freq_idx < len(deviation):
                local_deviation = deviation[freq_idx]
                if local_deviation < -self.peak_prominence:
                    gain = max(self.min_gain, local_deviation * sensitivity)
                    gain = min(0, gain)  # Only cuts
                    corrections.append(EQBand(
                        band_type='peak',
                        frequency=freq,
                        gain=gain * min(1.0, max_cut / abs(self.min_gain)),
                        q=q
                    ))
        
        # Apply profile's boost frequencies
        for freq, max_boost, q in profile.boost_frequencies:
            freq_idx = np.argmin(np.abs(freqs - freq))
            if freq_idx < len(deviation):
                local_deviation = deviation[freq_idx]
                if local_deviation > self.peak_prominence:
                    gain = min(self.max_gain, local_deviation * sensitivity)
                    gain = max(0, gain)  # Only boosts
                    corrections.append(EQBand(
                        band_type='peak',
                        frequency=freq,
                        gain=gain * min(1.0, max_boost / self.max_gain),
                        q=q
                    ))
        
        # Find additional problem frequencies (resonances to cut)
        peak_indices, peak_props = find_peaks(
            spectrum_smooth, 
            prominence=self.peak_prominence * 2,
            distance=20
        )
        
        for idx in peak_indices[:5]:  # Limit to top 5 resonances
            freq = freqs[idx]
            if 100 < freq < 10000:  # Only in mid-range
                prominence = peak_props['prominences'][list(peak_indices).index(idx)]
                if prominence > self.peak_prominence * 2:
                    # Calculate Q based on peak width
                    q = self._calculate_q_from_peak(spectrum_smooth, freqs, idx)
                    gain = max(self.min_gain, -prominence * 0.5 * sensitivity)
                    
                    # Don't add if too close to existing correction
                    too_close = any(abs(c.frequency - freq) < freq * 0.1 for c in corrections)
                    if not too_close:
                        corrections.append(EQBand(
                            band_type='peak',
                            frequency=freq,
                            gain=gain,
                            q=q
                        ))
        
        # Sort by frequency and limit to 6 bands (Wing EQ has 6 bands)
        corrections.sort(key=lambda x: x.frequency)
        
        # Assign to EQ bands
        if len(corrections) > 6:
            # Keep most significant corrections
            corrections.sort(key=lambda x: abs(x.gain), reverse=True)
            corrections = corrections[:6]
            corrections.sort(key=lambda x: x.frequency)
        
        # Assign band types based on position
        if corrections:
            corrections[0].band_type = 'low_shelf'
            if len(corrections) > 1:
                corrections[-1].band_type = 'high_shelf'
        
        return corrections
    
    def _calculate_q_from_peak(self, spectrum: np.ndarray, freqs: np.ndarray, peak_idx: int) -> float:
        """Calculate Q factor from peak width at -3dB."""
        peak_value = spectrum[peak_idx]
        threshold = peak_value - 3.0
        
        # Find left edge
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > threshold:
            left_idx -= 1
        
        # Find right edge
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > threshold:
            right_idx += 1
        
        # Calculate bandwidth
        if left_idx < right_idx and right_idx < len(freqs):
            bandwidth = freqs[right_idx] - freqs[left_idx]
            center_freq = freqs[peak_idx]
            if bandwidth > 0:
                q = center_freq / bandwidth
                return max(0.44, min(10.0, q))
        
        return 2.0  # Default Q
    
    def map_to_mixer_bands(self, corrections: List[EQBand]) -> Dict[str, Dict[str, float]]:
        """
        Map corrections to Wing mixer EQ band addresses.
        
        Wing EQ structure:
        - Low shelf: lg (gain), lf (freq), lq (Q)
        - Band 1-4: 1g/1f/1q, 2g/2f/2q, 3g/3f/3q, 4g/4f/4q
        - High shelf: hg (gain), hf (freq), hq (Q)
        
        Returns:
            Dict mapping band name to parameters
        """
        mixer_bands = {}
        
        if not corrections:
            return mixer_bands
        
        # Sort by frequency
        corrections_sorted = sorted(corrections, key=lambda x: x.frequency)
        
        # Assign to bands
        for i, correction in enumerate(corrections_sorted):
            if i == 0:
                # Low shelf
                mixer_bands['low'] = {
                    'lg': self._clamp_gain(correction.gain),
                    'lf': self._clamp_freq(correction.frequency, 20, 2000),
                    'lq': self._clamp_q(correction.q)
                }
            elif i == len(corrections_sorted) - 1 and len(corrections_sorted) > 1:
                # High shelf
                mixer_bands['high'] = {
                    'hg': self._clamp_gain(correction.gain),
                    'hf': self._clamp_freq(correction.frequency, 50, 20000),
                    'hq': self._clamp_q(correction.q)
                }
            else:
                # Parametric band (1-4)
                band_num = min(i, 4)
                mixer_bands[f'band{band_num}'] = {
                    f'{band_num}g': self._clamp_gain(correction.gain),
                    f'{band_num}f': self._clamp_freq(correction.frequency, 20, 20000),
                    f'{band_num}q': self._clamp_q(correction.q)
                }
        
        return mixer_bands
    
    def _clamp_gain(self, gain: float) -> float:
        """Clamp gain to Wing EQ range (-15 to +15 dB)."""
        return max(-15.0, min(15.0, gain))
    
    def _clamp_freq(self, freq: float, min_freq: float, max_freq: float) -> float:
        """Clamp frequency to valid range."""
        return max(min_freq, min(max_freq, freq))
    
    def _clamp_q(self, q: float) -> float:
        """Clamp Q to Wing EQ range (0.44 to 10)."""
        return max(0.44, min(10.0, q))


# ============================================================================
# EQ Preview (Pedalboard)
# ============================================================================

class EQPreview:
    """
    Local EQ preview using Spotify's Pedalboard library.
    
    Allows previewing EQ corrections on audio without sending to mixer.
    """
    
    def __init__(self, sample_rate: int = 48000):
        """
        Initialize EQ preview.
        
        Args:
            sample_rate: Audio sample rate in Hz
        """
        self.sample_rate = sample_rate
        self._pedalboard_available = False
        self._board = None
        
        self._init_pedalboard()
    
    def _init_pedalboard(self):
        """Initialize Pedalboard library."""
        try:
            from pedalboard import Pedalboard, HighShelfFilter, LowShelfFilter, PeakFilter
            self._pedalboard_available = True
            self._board = Pedalboard([])
            logger.info("Pedalboard initialized successfully")
        except ImportError as e:
            logger.warning(f"Pedalboard not available: {e}")
            self._pedalboard_available = False
    
    def apply_preview(self, audio: np.ndarray, corrections: List[EQBand]) -> np.ndarray:
        """
        Apply EQ corrections to audio for preview.
        
        Args:
            audio: Input audio samples (float32)
            corrections: List of EQ corrections to apply
            
        Returns:
            Processed audio samples
        """
        if not self._pedalboard_available or not corrections:
            return audio
        
        try:
            from pedalboard import Pedalboard, HighShelfFilter, LowShelfFilter, PeakFilter
            
            # Build effects chain
            effects = []
            
            for correction in corrections:
                if abs(correction.gain) < 0.5:
                    continue  # Skip negligible corrections
                
                if correction.band_type == 'low_shelf':
                    effects.append(LowShelfFilter(
                        cutoff_frequency_hz=correction.frequency,
                        gain_db=correction.gain,
                        q=correction.q
                    ))
                elif correction.band_type == 'high_shelf':
                    effects.append(HighShelfFilter(
                        cutoff_frequency_hz=correction.frequency,
                        gain_db=correction.gain,
                        q=correction.q
                    ))
                else:  # peak
                    effects.append(PeakFilter(
                        cutoff_frequency_hz=correction.frequency,
                        gain_db=correction.gain,
                        q=correction.q
                    ))
            
            if not effects:
                return audio
            
            # Create pedalboard and process
            board = Pedalboard(effects)
            
            # Ensure audio is 2D (channels, samples)
            if audio.ndim == 1:
                audio = audio.reshape(1, -1)
            
            processed = board(audio.astype(np.float32), self.sample_rate)
            
            # Return same shape as input
            if processed.shape[0] == 1:
                return processed.flatten()
            return processed
            
        except Exception as e:
            logger.error(f"Error applying EQ preview: {e}")
            return audio
    
    def is_available(self) -> bool:
        """Check if Pedalboard preview is available."""
        return self._pedalboard_available


# Spectral Gate removed - now using centralized bleed_service from auto_fader_v2/core

# ============================================================================
# Auto EQ Controller
# ============================================================================

class AutoEQController:
    """
    Main controller for Auto-EQ functionality.
    
    Coordinates spectrum analysis, correction calculation, preview, and mixer control.
    """
    
    def __init__(self, mixer_client=None, dominance_threshold: float = 0.7, min_analysis_time: float = 5.0, bleed_service=None):
        """
        Initialize Auto-EQ controller.
        
        Args:
            mixer_client: WingClient instance for sending commands to mixer
            dominance_threshold: Legacy parameter (kept for compatibility, not used)
            min_analysis_time: Minimum analysis time in seconds before applying corrections
            bleed_service: Centralized bleed detection service
        """
        self.mixer_client = mixer_client
        
        # Components
        self.analyzer = None
        self.corrector = EQCorrector()
        self.preview = EQPreview()
        self.bleed_service = bleed_service
        
        # State
        self.is_active = False
        self.current_channel: Optional[int] = None
        self.current_profile: Optional[InstrumentProfile] = None
        self.current_corrections: List[EQBand] = []
        self.auto_apply = False  # Legacy; use apply_mode
        self.apply_mode: str = "suggest"  # "auto" | "suggest" | "confirm" — suggest = only show, confirm = apply on user confirm, auto = apply immediately
        
        # Spectral gating and buffering
        self.monitored_channels: List[int] = []  # List of channel IDs to monitor for bleeding
        self.spectrum_buffer: deque = deque(maxlen=int(min_analysis_time * 30))  # 5 sec * 30 fps
        self.min_analysis_time = min_analysis_time
        self.analysis_start_time: Optional[float] = None
        
        # Correction history tracking
        self.applied_corrections_history: List[List[EQBand]] = []
        self.cumulative_eq_response: Optional[np.ndarray] = None
        
        # Callbacks
        self.on_spectrum_update: Optional[Callable[[Dict], None]] = None
        self.on_corrections_calculated: Optional[Callable[[List[Dict]], None]] = None
        self.on_status_update: Optional[Callable[[Dict], None]] = None
        
        logger.info("AutoEQController initialized")
    
    def start(self, 
              device_id: int,
              channel: int,
              profile_name: str = 'custom',
              auto_apply: bool = False,
              apply_mode: str = None,  # "auto" | "suggest" | "confirm", overrides auto_apply if set
              monitored_channels: List[int] = None,
              on_spectrum_callback: Callable = None,
              on_corrections_callback: Callable = None,
              on_status_callback: Callable = None) -> bool:
        """
        Start auto-EQ analysis for a channel.
        
        Args:
            device_id: PyAudio device index for audio input
            channel: Mixer channel number (1-40)
            profile_name: Instrument profile name
            auto_apply: Automatically apply corrections to mixer
            monitored_channels: List of channel IDs to monitor for bleeding rejection
            on_spectrum_callback: Callback for spectrum updates
            on_corrections_callback: Callback for correction updates
            on_status_callback: Callback for status updates
            
        Returns:
            True if started successfully
        """
        if self.is_active:
            logger.warning("Auto-EQ already active")
            return False
        
        self.current_channel = channel
        self.current_profile = InstrumentProfiles.get_profile(profile_name)
        self.auto_apply = auto_apply
        if apply_mode is not None:
            self.apply_mode = apply_mode
        else:
            self.apply_mode = "auto" if auto_apply else "suggest"
        self.monitored_channels = monitored_channels or []
        
        # Reset analysis state
        self.spectrum_buffer.clear()
        self.analysis_start_time = None
        self.applied_corrections_history.clear()
        self.cumulative_eq_response = None
        
        self.on_spectrum_update = on_spectrum_callback
        self.on_corrections_calculated = on_corrections_callback
        self.on_status_update = on_status_callback
        
        # Initialize analyzer
        self.analyzer = SpectrumAnalyzer(device_index=device_id)
        
        # Start analysis
        success = self.analyzer.start(
            channel=channel,
            on_spectrum_callback=self._on_spectrum_received
        )
        
        if success:
            self.is_active = True
            self.analysis_start_time = time.time()
            self._notify_status('started', f'Auto-EQ started for channel {channel}')
            logger.info(f"Auto-EQ started: channel {channel}, profile {profile_name}, monitored_channels={self.monitored_channels}")
        else:
            self._notify_status('error', 'Failed to start spectrum analyzer')
        
        return success
    
    def stop(self):
        """Stop auto-EQ analysis."""
        self.is_active = False
        
        if self.analyzer:
            self.analyzer.stop()
            self.analyzer = None
        
        self._notify_status('stopped', 'Auto-EQ stopped')
        logger.info("Auto-EQ stopped")
    
    def set_profile(self, profile_name: str):
        """
        Change the instrument profile.
        
        Args:
            profile_name: New profile name
        """
        self.current_profile = InstrumentProfiles.get_profile(profile_name)
        logger.info(f"Profile changed to: {profile_name}")
        
        # Recalculate corrections with current spectrum
        if self.analyzer and self.analyzer.current_spectrum:
            self._calculate_and_notify_corrections(self.analyzer.current_spectrum)
    
    def set_channel(self, channel: int):
        """
        Change the target mixer channel.
        
        Args:
            channel: New channel number (1-40)
        """
        self.current_channel = channel
        logger.info(f"Channel changed to: {channel}")
    
    def set_auto_apply(self, enabled: bool):
        """Enable/disable automatic correction application (sets apply_mode to 'auto' or 'suggest')."""
        self.auto_apply = enabled
        self.apply_mode = "auto" if enabled else "suggest"
        logger.info(f"Auto-apply: {enabled}, apply_mode: {self.apply_mode}")
    
    def _on_spectrum_received(self, spectral_data: SpectralData):
        """Handle incoming spectrum data."""
        if not self.is_active:
            return
        
        # Bleed detection now handled by centralized bleed_service
        # Use original spectrum for analysis (bleed compensation applied in correction calculation)
        gated_spectrum = spectral_data.spectrum
        
        # Create gated spectral data
        gated_spectral_data = SpectralData(
            spectrum=gated_spectrum,
            frequencies=spectral_data.frequencies,
            peak_freq=spectral_data.peak_freq,
            centroid=spectral_data.centroid,
            rolloff=spectral_data.rolloff,
            flatness=spectral_data.flatness,
            bandwidth=spectral_data.bandwidth,
            peaks=spectral_data.peaks
        )
        
        # Add to buffer
        self.spectrum_buffer.append(gated_spectral_data)
        
        # Check if we have enough data accumulated
        elapsed_time = time.time() - self.analysis_start_time if self.analysis_start_time else 0
        
        # Send spectrum for visualization (without delay)
        if self.on_spectrum_update:
            spectrum_viz = self.analyzer.get_spectrum_for_visualization(32)
            target_curve = self._get_target_curve_for_visualization()
            self.on_spectrum_update({
                'spectrum': spectrum_viz,
                'target_curve': target_curve,
                'peak_freq': spectral_data.peak_freq,
                'centroid': spectral_data.centroid,
                'rolloff': spectral_data.rolloff,
                'flatness': spectral_data.flatness
            })
        
        # Calculate corrections only after accumulating enough data
        if (elapsed_time >= self.min_analysis_time and 
            len(self.spectrum_buffer) >= 10 and 
            self.current_profile):
            averaged_spectrum = self._get_averaged_spectrum()
            if averaged_spectrum:
                self._calculate_and_notify_corrections(averaged_spectrum)
    
    def _get_averaged_spectrum(self) -> Optional[SpectralData]:
        """
        Get averaged spectrum from buffer.
        
        Returns:
            Averaged SpectralData or None if buffer is empty
        """
        if len(self.spectrum_buffer) == 0:
            return None
        
        # Average all spectra in buffer
        spectra = [sd.spectrum for sd in self.spectrum_buffer]
        averaged_spectrum = np.mean(spectra, axis=0)
        
        # Use frequencies from the most recent spectrum
        latest = self.spectrum_buffer[-1]
        
        # Recalculate features from averaged spectrum
        # (simplified - using latest features)
        return SpectralData(
            spectrum=averaged_spectrum,
            frequencies=latest.frequencies,
            peak_freq=latest.peak_freq,
            centroid=latest.centroid,
            rolloff=latest.rolloff,
            flatness=latest.flatness,
            bandwidth=latest.bandwidth,
            peaks=latest.peaks
        )
    
    def _get_target_curve_for_visualization(self) -> List[float]:
        """
        Get target curve data for visualization.
        
        Returns:
            List of dB values for visualization bands
        """
        if not self.current_profile:
            return []
        
        # Interpolate target curve to visualization frequency bands
        num_bands = 32
        band_edges = np.logspace(np.log10(20), np.log10(20000), num_bands + 1)
        band_centers = [(band_edges[i] + band_edges[i+1]) / 2 for i in range(num_bands)]
        
        # Interpolate target curve
        target_freqs = [p[0] for p in self.current_profile.target_curve]
        target_gains = [p[1] for p in self.current_profile.target_curve]
        
        if len(target_freqs) < 2:
            return [0.0] * num_bands
        
        try:
            from scipy.interpolate import interp1d
            interp_func = interp1d(
                target_freqs, 
                target_gains,
                kind='linear',
                bounds_error=False,
                fill_value=(target_gains[0], target_gains[-1])
            )
            target_curve = [float(interp_func(freq)) for freq in band_centers]
            return target_curve
        except Exception as e:
            logger.error(f"Error interpolating target curve: {e}")
            return [0.0] * num_bands
    
    def _apply_cumulative_eq_to_spectrum(self, spectrum: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """
        Apply inverse of cumulative EQ to spectrum to compensate for already applied corrections.
        
        Args:
            spectrum: Input spectrum
            frequencies: Frequency array
            
        Returns:
            Compensated spectrum
        """
        if not self.applied_corrections_history or len(self.applied_corrections_history) == 0:
            return spectrum
        
        # Calculate cumulative EQ response
        # For now, we'll use a simplified approach
        # In a full implementation, we'd use EQPreview to simulate the cumulative effect
        
        # Simple compensation: reduce gain at frequencies where we've boosted
        compensated_spectrum = spectrum.copy()
        
        # Get all applied corrections
        all_corrections = []
        for corrections_list in self.applied_corrections_history:
            all_corrections.extend(corrections_list)
        
        # Apply inverse corrections (simplified)
        for correction in all_corrections:
            # Find frequency index
            freq_idx = np.argmin(np.abs(frequencies - correction.frequency))
            if freq_idx < len(compensated_spectrum):
                # Compensate by reducing the effect
                # This is a simplified model - full implementation would use proper EQ simulation
                compensation_factor = 1.0 - (correction.gain / 20.0) * 0.1  # Small compensation
                compensated_spectrum[freq_idx] *= max(0.1, compensation_factor)
        
        return compensated_spectrum
    
    def _calculate_and_notify_corrections(self, spectral_data: SpectralData):
        """Calculate corrections and notify callbacks."""
        # Apply cumulative EQ compensation if we have applied corrections
        compensated_spectrum = spectral_data.spectrum
        if self.applied_corrections_history:
            compensated_spectrum = self._apply_cumulative_eq_to_spectrum(
                spectral_data.spectrum,
                spectral_data.frequencies
            )
        
        # Create compensated spectral data
        compensated_spectral_data = SpectralData(
            spectrum=compensated_spectrum,
            frequencies=spectral_data.frequencies,
            peak_freq=spectral_data.peak_freq,
            centroid=spectral_data.centroid,
            rolloff=spectral_data.rolloff,
            flatness=spectral_data.flatness,
            bandwidth=spectral_data.bandwidth,
            peaks=spectral_data.peaks
        )
        
        corrections = self.corrector.calculate_correction(
            compensated_spectral_data, 
            self.current_profile,
            sensitivity=1.0
        )
        
        # Apply bleed compensation: reduce boost corrections when bleed is high
        if self.bleed_service and self.bleed_service.enabled and self.current_channel:
            bleed_info = self.bleed_service.get_bleed_info(self.current_channel)
            if bleed_info and bleed_info.bleed_ratio > 0:
                # Scale down boost corrections proportionally to bleed ratio
                bleed_factor = 1.0 - (bleed_info.bleed_ratio * 0.5)  # Max 50% reduction
                bleed_factor = max(0.3, bleed_factor)  # Never reduce below 30%
                for correction in corrections:
                    if correction.gain > 0:  # Only reduce boosts, not cuts
                        correction.gain *= bleed_factor
                        logger.debug(f"Ch{self.current_channel}: Scaled boost correction at {correction.frequency} Hz "
                                   f"by {bleed_factor:.2f} due to bleed (ratio={bleed_info.bleed_ratio:.2f})")
        
        self.current_corrections = corrections
        
        # Notify callback
        if self.on_corrections_calculated:
            corrections_dict = [c.to_dict() for c in corrections]
            self.on_corrections_calculated(corrections_dict)
        
        # Apply according to mode: auto = apply now; suggest/confirm = only notify (user applies for confirm)
        if self.apply_mode == "auto" and corrections:
            self.apply_to_mixer()
    
    def apply_to_mixer(self, corrections: List[EQBand] = None) -> bool:
        """
        Apply EQ corrections to the mixer.
        
        Args:
            corrections: Optional corrections to apply (uses current if None)
            
        Returns:
            True if applied successfully
        """
        if not self.mixer_client or not self.mixer_client.is_connected:
            logger.warning("Mixer not connected")
            self._notify_status('error', 'Mixer not connected')
            return False
        
        if not self.current_channel:
            logger.warning("No channel selected")
            return False
        
        corrections = corrections or self.current_corrections
        if not corrections:
            logger.debug("No corrections to apply (signal may be too quiet or already matches target)")
            return False
        
        try:
            # Enable EQ
            self.mixer_client.set_eq_on(self.current_channel, 1)
            
            # Map corrections to mixer bands
            mixer_bands = self.corrector.map_to_mixer_bands(corrections)
            
            # Apply each band
            for band_name, params in mixer_bands.items():
                if band_name == 'low':
                    self.mixer_client.set_eq_low_shelf(
                        self.current_channel,
                        gain=params.get('lg'),
                        freq=params.get('lf'),
                        q=params.get('lq')
                    )
                elif band_name == 'high':
                    self.mixer_client.set_eq_high_shelf(
                        self.current_channel,
                        gain=params.get('hg'),
                        freq=params.get('hf'),
                        q=params.get('hq')
                    )
                elif band_name.startswith('band'):
                    band_num = int(band_name[-1])
                    self.mixer_client.set_eq_band(
                        self.current_channel,
                        band=band_num,
                        freq=params.get(f'{band_num}f'),
                        gain=params.get(f'{band_num}g'),
                        q=params.get(f'{band_num}q')
                    )
            
            # Store applied corrections in history
            self.applied_corrections_history.append(corrections.copy())
            
            # Limit history size (keep last 10 applications)
            if len(self.applied_corrections_history) > 10:
                self.applied_corrections_history.pop(0)
            
            logger.info(f"Applied EQ corrections to channel {self.current_channel}")
            self._notify_status('applied', f'EQ applied to channel {self.current_channel}')
            return True
            
        except Exception as e:
            logger.error(f"Error applying EQ: {e}")
            self._notify_status('error', f'Error applying EQ: {e}')
            return False
    
    def reset_eq(self) -> bool:
        """
        Reset EQ to flat (0 dB on all bands).
        
        Returns:
            True if reset successfully
        """
        if not self.mixer_client or not self.mixer_client.is_connected:
            logger.warning("Mixer not connected")
            return False
        
        if not self.current_channel:
            return False
        
        try:
            ch = self.current_channel
            
            # Ensure EQ is enabled (needed for changes to take effect)
            if hasattr(self.mixer_client, 'set_eq_on'):
                self.mixer_client.set_eq_on(ch, 1)
                import time
                time.sleep(0.05)
            
            # Reset all bands to 0 dB using direct OSC addresses (only reset gain, don't change Q or frequency)
            import time
            logger.info(f"Reset EQ: Sending gain=0 commands for channel {ch}")
            
            # Low shelf gain
            self.mixer_client.send(f"/ch/{ch}/eq/lg", 0.0)
            time.sleep(0.05)
            
            # Band 1-4 gains
            for band in [1, 2, 3, 4]:
                self.mixer_client.send(f"/ch/{ch}/eq/{band}g", 0.0)
                time.sleep(0.05)
            
            # High shelf gain
            self.mixer_client.send(f"/ch/{ch}/eq/hg", 0.0)
            
            self.current_corrections = []
            
            logger.info(f"Reset EQ for channel {ch}")
            self._notify_status('reset', f'EQ reset for channel {ch}')
            return True
            
        except Exception as e:
            logger.error(f"Error resetting EQ: {e}")
            return False
    
    def get_current_eq(self) -> Optional[Dict[str, Any]]:
        """
        Get current EQ settings from mixer.
        
        Returns:
            Dict with current EQ parameters
        """
        if not self.mixer_client or not self.current_channel:
            return None
        
        try:
            ch = self.current_channel
            
            # Query all EQ parameters
            eq_data = {
                'channel': ch,
                'on': self.mixer_client.get_eq_on(ch),
                'low_shelf': {
                    'gain': self.mixer_client.get_eq_band_gain(ch, 'lg'),
                    'freq': self.mixer_client.get_eq_band_frequency(ch, 'lf'),
                },
                'bands': [],
                'high_shelf': {
                    'gain': self.mixer_client.get_eq_band_gain(ch, 'hg'),
                    'freq': self.mixer_client.get_eq_band_frequency(ch, 'hf'),
                }
            }
            
            for band in range(1, 5):
                eq_data['bands'].append({
                    'band': band,
                    'gain': self.mixer_client.get_eq_band_gain(ch, f'{band}g'),
                    'freq': self.mixer_client.get_eq_band_frequency(ch, f'{band}f'),
                })
            
            return eq_data
            
        except Exception as e:
            logger.error(f"Error getting EQ: {e}")
            return None
    
    def preview_corrections(self, audio: np.ndarray) -> np.ndarray:
        """
        Preview EQ corrections on audio buffer.
        
        Args:
            audio: Input audio samples
            
        Returns:
            Processed audio with EQ applied
        """
        if not self.current_corrections:
            return audio
        
        return self.preview.apply_preview(audio, self.current_corrections)
    
    def _notify_status(self, status_type: str, message: str):
        """Send status update to callback."""
        if self.on_status_update:
            self.on_status_update({
                'type': status_type,
                'message': message,
                'channel': self.current_channel,
                'profile': self.current_profile.name if self.current_profile else None,
                'active': self.is_active
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current controller status."""
        return {
            'active': self.is_active,
            'channel': self.current_channel,
            'profile': self.current_profile.name if self.current_profile else None,
            'auto_apply': self.auto_apply,
            'apply_mode': self.apply_mode,
            'corrections_count': len(self.current_corrections),
            'preview_available': self.preview.is_available()
        }


def room_analysis_stub(sample_rate: int = 48000) -> Dict[str, Any]:
    """Placeholder for Room Analysis (RTA with pink noise). Not implemented; returns stub result."""
    return {"implemented": False, "message": "Room analysis (RTA) not implemented", "sample_rate": sample_rate}


def resolve_frequency_conflicts(corrections_per_channel: Dict[int, List[EQBand]], low_band_freq_range: Tuple[float, float] = (100.0, 200.0)) -> Dict[int, List[EQBand]]:
    """
    Reduce masking when multiple channels boost the same low band (e.g. kick + bass 100-200 Hz).
    Returns a copy of corrections with gains reduced on conflicting bands where multiple channels boost.
    """
    result = {ch: list(corrections) for ch, corrections in corrections_per_channel.items()}
    low_min, low_max = low_band_freq_range
    boosting_channels = []
    for ch, bands in result.items():
        for b in bands:
            if low_min <= b.frequency <= low_max and b.gain > 0:
                boosting_channels.append((ch, b))
                break
    if len(boosting_channels) < 2:
        return result
    for ch, _ in boosting_channels[1:]:
        for i, b in enumerate(result[ch]):
            if low_min <= b.frequency <= low_max and b.gain > 0:
                new_gain = b.gain * 0.5
                result[ch][i] = EQBand(band_type=b.band_type, frequency=b.frequency, gain=new_gain, q=b.q)
                logger.info(f"Conflict resolver: reduced Ch{ch} {b.frequency} Hz boost to {new_gain:.1f} dB")
                break
    return result


# ============================================================================
# Multi-Channel Auto EQ Controller
# ============================================================================

class MultiChannelAutoEQController:
    """
    Controller for mass auto-EQ processing of multiple channels simultaneously.
    
    Manages multiple SpectrumAnalyzer instances (one per channel) and applies
    spectral gating to filter bleeding between channels.
    """
    
    def __init__(self, mixer_client=None, dominance_threshold: float = 0.7, min_analysis_time: float = 5.0, bleed_service=None):
        """
        Initialize multi-channel Auto-EQ controller.
        
        Args:
            mixer_client: WingClient instance for sending commands to mixer
            dominance_threshold: Legacy parameter (kept for compatibility, not used)
            min_analysis_time: Minimum analysis time in seconds before applying corrections
            bleed_service: Centralized bleed detection service
        """
        self.mixer_client = mixer_client
        
        # Centralized bleed detection service
        self.bleed_service = bleed_service
        self.min_analysis_time = min_analysis_time
        
        # Active channels: Dict[channel_id, channel_state]
        self.active_channels: Dict[int, Dict] = {}
        
        # Components
        self.corrector = EQCorrector()
        self.preview = EQPreview()
        
        # Callbacks
        self.on_spectrum_update: Optional[Callable[[Dict], None]] = None
        self.on_corrections_calculated: Optional[Callable[[Dict], None]] = None
        self.on_status_update: Optional[Callable[[Dict], None]] = None
        
        logger.info("MultiChannelAutoEQController initialized")
    
    def start_multi_channel(self, 
                           device_id: int,
                           channels_config: List[Dict],
                           on_spectrum_callback: Callable = None,
                           on_corrections_callback: Callable = None,
                           on_status_callback: Callable = None) -> bool:
        """
        Start auto-EQ analysis for multiple channels.
        
        Args:
            device_id: PyAudio device index for audio input
            channels_config: List of channel configs, each with:
                - channel: int (mixer channel number)
                - profile: str (profile name)
                - auto_apply: bool
            on_spectrum_callback: Callback for spectrum updates (receives {channel, spectrum, ...})
            on_corrections_callback: Callback for correction updates (receives {channel, corrections})
            on_status_callback: Callback for status updates
            
        Returns:
            True if started successfully
        """
        if len(self.active_channels) > 0:
            logger.warning("Multi-channel Auto-EQ already active, stopping existing")
            self.stop_all()
        
        self.on_spectrum_update = on_spectrum_callback
        self.on_corrections_calculated = on_corrections_callback
        self.on_status_update = on_status_callback
        
        # Collect all channel IDs for spectral gate monitoring
        all_channel_ids = [config['channel'] for config in channels_config]
        
        # Start analyzer for each channel
        started_count = 0
        for config in channels_config:
            channel = config['channel']
            profile_name = config.get('profile', 'custom')
            auto_apply = config.get('auto_apply', False)
            
            logger.info(f"Starting analyzer for channel {channel} with profile {profile_name}, device_id={device_id}")
            
            # Create analyzer for this channel
            analyzer = SpectrumAnalyzer(device_index=device_id)
            
            # Create callback for this channel (use closure to capture channel value)
            def make_channel_callback(ch):
                def on_spectrum(spectral_data):
                    self._on_channel_spectrum(ch, spectral_data)
                return on_spectrum
            
            # Start analyzer
            try:
                success = analyzer.start(
                    channel=channel,
                    on_spectrum_callback=make_channel_callback(channel)
                )
                
                if success:
                    # Store channel state
                    profile = InstrumentProfiles.get_profile(profile_name)
                    if not profile:
                        logger.warning(f"Profile '{profile_name}' not found, using 'custom'")
                        profile = InstrumentProfiles.get_profile('custom')
                    
                    self.active_channels[channel] = {
                        'analyzer': analyzer,
                        'profile': profile,
                        'auto_apply': auto_apply,
                        'corrections': [],
                        'spectrum_buffer': deque(maxlen=int(self.min_analysis_time * 30)),
                        'analysis_start_time': time.time(),
                        'device_id': device_id
                    }
                    started_count += 1
                    logger.info(f"Successfully started analysis for channel {channel} with profile {profile_name}")
                else:
                    logger.error(f"Failed to start analyzer for channel {channel}")
            except Exception as e:
                logger.error(f"Exception starting analyzer for channel {channel}: {e}", exc_info=True)
        
        if started_count > 0:
            logger.info(f"Multi-channel Auto-EQ started for {started_count}/{len(channels_config)} channels")
            return True
        else:
            logger.error(f"Failed to start any channel analyzers out of {len(channels_config)} requested")
            return False
    
    def _on_channel_spectrum(self, channel: int, spectral_data: SpectralData):
        """Handle spectrum data for a specific channel."""
        if channel not in self.active_channels:
            return
        
        channel_state = self.active_channels[channel]
        
        # Bleed detection now handled by centralized bleed_service
        # Use original spectrum for analysis (bleed compensation applied in correction calculation)
        gated_spectrum = spectral_data.spectrum
        
        # Create gated spectral data
        gated_spectral_data = SpectralData(
            spectrum=gated_spectrum,
            frequencies=spectral_data.frequencies,
            peak_freq=spectral_data.peak_freq,
            centroid=spectral_data.centroid,
            rolloff=spectral_data.rolloff,
            flatness=spectral_data.flatness,
            bandwidth=spectral_data.bandwidth,
            peaks=spectral_data.peaks
        )
        
        # Add to buffer
        channel_state['spectrum_buffer'].append(gated_spectral_data)
        
        # Check if we have enough data
        elapsed_time = time.time() - channel_state['analysis_start_time']
        
        # Send spectrum for visualization
        if self.on_spectrum_update:
            spectrum_viz = channel_state['analyzer'].get_spectrum_for_visualization(32)
            target_curve = self._get_target_curve_for_channel(channel)
            self.on_spectrum_update({
                'channel': channel,
                'spectrum': spectrum_viz,
                'target_curve': target_curve,
                'peak_freq': spectral_data.peak_freq,
                'centroid': spectral_data.centroid,
                'rolloff': spectral_data.rolloff,
                'flatness': spectral_data.flatness
            })
        
        # Calculate corrections after accumulating data
        if (elapsed_time >= self.min_analysis_time and 
            len(channel_state['spectrum_buffer']) >= 10 and 
            channel_state['profile']):
            averaged_spectrum = self._get_averaged_spectrum_for_channel(channel)
            if averaged_spectrum:
                self._calculate_and_notify_corrections(channel, averaged_spectrum)
    
    def _get_averaged_spectrum_for_channel(self, channel: int) -> Optional[SpectralData]:
        """Get averaged spectrum from buffer for a channel."""
        if channel not in self.active_channels:
            return None
        
        buffer = self.active_channels[channel]['spectrum_buffer']
        if len(buffer) == 0:
            return None
        
        # Average all spectra in buffer
        spectra = [sd.spectrum for sd in buffer]
        averaged_spectrum = np.mean(spectra, axis=0)
        
        # Use frequencies from the most recent spectrum
        latest = buffer[-1]
        
        return SpectralData(
            spectrum=averaged_spectrum,
            frequencies=latest.frequencies,
            peak_freq=latest.peak_freq,
            centroid=latest.centroid,
            rolloff=latest.rolloff,
            flatness=latest.flatness,
            bandwidth=latest.bandwidth,
            peaks=latest.peaks
        )
    
    def _get_target_curve_for_channel(self, channel: int) -> List[float]:
        """Get target curve for visualization for a channel."""
        if channel not in self.active_channels:
            return []
        
        profile = self.active_channels[channel]['profile']
        if not profile:
            return []
        
        # Interpolate target curve to visualization frequency bands
        num_bands = 32
        band_edges = np.logspace(np.log10(20), np.log10(20000), num_bands + 1)
        band_centers = [(band_edges[i] + band_edges[i+1]) / 2 for i in range(num_bands)]
        
        # Interpolate target curve
        target_freqs = [p[0] for p in profile.target_curve]
        target_gains = [p[1] for p in profile.target_curve]
        
        if len(target_freqs) < 2:
            return [0.0] * num_bands
        
        try:
            from scipy.interpolate import interp1d
            interp_func = interp1d(
                target_freqs, 
                target_gains,
                kind='linear',
                bounds_error=False,
                fill_value=(target_gains[0], target_gains[-1])
            )
            target_curve = [float(interp_func(freq)) for freq in band_centers]
            return target_curve
        except Exception as e:
            logger.error(f"Error interpolating target curve: {e}")
            return [0.0] * num_bands
    
    def _calculate_and_notify_corrections(self, channel: int, spectral_data: SpectralData):
        """Calculate corrections for a channel and notify callbacks."""
        if channel not in self.active_channels:
            return
        
        channel_state = self.active_channels[channel]
        profile = channel_state['profile']
        
        if not profile:
            return
        
        # Calculate corrections
        corrections = self.corrector.calculate_correction(
            spectral_data,
            profile,
            sensitivity=1.0
        )
        
        # Apply bleed compensation: reduce boost corrections when bleed is high
        if self.bleed_service and self.bleed_service.enabled:
            bleed_info = self.bleed_service.get_bleed_info(channel)
            if bleed_info and bleed_info.bleed_ratio > 0:
                # Scale down boost corrections proportionally to bleed ratio
                bleed_factor = 1.0 - (bleed_info.bleed_ratio * 0.5)  # Max 50% reduction
                bleed_factor = max(0.3, bleed_factor)  # Never reduce below 30%
                for correction in corrections:
                    if correction.gain > 0:  # Only reduce boosts, not cuts
                        correction.gain *= bleed_factor
                        logger.debug(f"Ch{channel}: Scaled boost correction at {correction.frequency} Hz "
                                   f"by {bleed_factor:.2f} due to bleed (ratio={bleed_info.bleed_ratio:.2f})")
        
        channel_state['corrections'] = corrections
        
        # Notify callback
        if self.on_corrections_calculated:
            corrections_dict = [c.to_dict() for c in corrections]
            self.on_corrections_calculated({
                'channel': channel,
                'corrections': corrections_dict
            })
        
        # Auto-apply if enabled
        if channel_state['auto_apply'] and corrections:
            self.apply_channel_correction(channel)
    
    def set_channel_profile(self, channel: int, profile_name: str):
        """Change profile for a specific channel."""
        if channel not in self.active_channels:
            logger.warning(f"Channel {channel} not found in active channels")
            return
        
        profile = InstrumentProfiles.get_profile(profile_name)
        self.active_channels[channel]['profile'] = profile
        
        # Reset analysis for this channel
        self.active_channels[channel]['spectrum_buffer'].clear()
        self.active_channels[channel]['analysis_start_time'] = time.time()
        
        logger.info(f"Profile changed for channel {channel} to {profile_name}")
    
    def apply_channel_correction(self, channel: int) -> bool:
        """Apply corrections for a specific channel."""
        # Check if channel has corrections (even if analysis stopped)
        if channel not in self.active_channels:
            logger.warning(f"Channel {channel} not found in active channels")
            return False
        
        if not self.mixer_client or not self.mixer_client.is_connected:
            logger.warning("Mixer not connected")
            return False
        
        corrections = self.active_channels[channel].get('corrections', [])
        if not corrections:
            logger.debug(f"No corrections to apply for channel {channel} (signal may be too quiet or already matches target)")
            return False
        
        try:
            # Enable EQ
            self.mixer_client.set_eq_on(channel, 1)
            
            # Map corrections to mixer bands
            mixer_bands = self.corrector.map_to_mixer_bands(corrections)
            
            # Apply each band
            for band_name, params in mixer_bands.items():
                if band_name == 'low':
                    self.mixer_client.set_eq_low_shelf(
                        channel,
                        gain=params.get('lg'),
                        freq=params.get('lf'),
                        q=params.get('lq')
                    )
                elif band_name == 'high':
                    self.mixer_client.set_eq_high_shelf(
                        channel,
                        gain=params.get('hg'),
                        freq=params.get('hf'),
                        q=params.get('hq')
                    )
                elif band_name.startswith('band'):
                    band_num = int(band_name[-1])
                    self.mixer_client.set_eq_band(
                        channel,
                        band=band_num,
                        freq=params.get(f'{band_num}f'),
                        gain=params.get(f'{band_num}g'),
                        q=params.get(f'{band_num}q')
                    )
            
            logger.info(f"Applied EQ corrections to channel {channel}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying EQ to channel {channel}: {e}")
            return False
    
    def apply_all_corrections(self, use_conflict_resolver: bool = True) -> Dict[int, bool]:
        """Apply corrections for all channels. If use_conflict_resolver, reduces overlapping low-band boosts (e.g. kick+bass 100-200 Hz)."""
        if use_conflict_resolver:
            corrections_per_ch = {
                ch: state.get('corrections', [])
                for ch, state in self.active_channels.items()
                if state.get('corrections')
            }
            if len(corrections_per_ch) >= 2:
                resolved = resolve_frequency_conflicts(corrections_per_ch)
                for ch, bands in resolved.items():
                    if ch in self.active_channels:
                        self.active_channels[ch]['corrections'] = bands
        results = {}
        for channel in list(self.active_channels.keys()):
            results[channel] = self.apply_channel_correction(channel)
        return results
    
    def stop_all(self):
        """Stop analysis for all channels."""
        # Save corrections before clearing
        saved_corrections = {}
        for channel, state in list(self.active_channels.items()):
            if state.get('corrections'):
                saved_corrections[channel] = state['corrections']
            if state['analyzer']:
                state['analyzer'].stop()
        
        # Clear active channels but keep corrections in a separate storage
        # This allows applying corrections after analysis stops
        self.active_channels.clear()
        
        # Store saved corrections for later use
        if saved_corrections:
            for channel, corrections in saved_corrections.items():
                # Recreate channel state with saved corrections (without analyzer)
                self.active_channels[channel] = {
                    'analyzer': None,
                    'profile': None,  # Profile info lost, but corrections remain
                    'auto_apply': False,
                    'corrections': corrections,
                    'spectrum_buffer': deque(),
                    'analysis_start_time': None,
                    'device_id': None
                }
        
        logger.info(f"Multi-channel Auto-EQ stopped. Saved corrections for {len(saved_corrections)} channels.")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status for all channels."""
        return {
            'active': len(self.active_channels) > 0,
            'channels': {
                ch: {
                    'profile': state['profile'].name if state['profile'] else None,
                    'auto_apply': state['auto_apply'],
                    'corrections_count': len(state['corrections'])
                }
                for ch, state in self.active_channels.items()
            }
        }
