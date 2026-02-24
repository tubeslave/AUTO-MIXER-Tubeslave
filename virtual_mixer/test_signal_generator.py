"""
Test Signal Generator for Virtual Mixer

Generates realistic audio signals for testing:
- Sine waves (fundamental frequencies)
- Pink noise
- Drum hits (kick, snare)
- Vocals (vowel synthesis)
- Mixed signals

Usage:
    python test_signal_generator.py --channels 32 --duration 60
"""

import numpy as np
import argparse
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of test signals."""
    SILENCE = "silence"
    SINE = "sine"
    PINK_NOISE = "pink_noise"
    WHITE_NOISE = "white_noise"
    DRUM_HIT = "drum_hit"
    VOCAL_AHH = "vocal_ahh"
    MIXED = "mixed"


@dataclass
class ChannelSignal:
    """Signal configuration for a channel."""
    ch_num: int
    signal_type: SignalType
    frequency: float = 1000.0  # For sine/vocal
    level_db: float = -20.0
    rhythm: float = 1.0  # Hz for drum hits
    onsets: List[float] = None  # Manual onset times


class TestSignalGenerator:
    """
    Generates test audio signals for virtual mixer.
    
    Simulates realistic signals:
    - Kick: 60Hz sine with fast attack
    - Snare: 200Hz + noise burst
    - Vocals: Vowel formants
    - Bass: 100Hz sustained
    - Keys: Multiple harmonics
    """
    
    SAMPLE_RATE = 48000
    
    def __init__(self):
        self.channels: Dict[int, ChannelSignal] = {}
        self.sample_rate = self.SAMPLE_RATE
        
        # Setup default signals for typical channels
        self._setup_default_signals()
        
        logger.info(f"TestSignalGenerator initialized")
    
    def _setup_default_signals(self):
        """Setup default signals for channels."""
        # Drums
        self.channels[1] = ChannelSignal(1, SignalType.DRUM_HIT, 60, -10, 2.0)   # Kick
        self.channels[2] = ChannelSignal(2, SignalType.DRUM_HIT, 80, -12, 2.0)   # Kick out
        self.channels[3] = ChannelSignal(3, SignalType.DRUM_HIT, 200, -8, 2.5)   # Snare
        self.channels[4] = ChannelSignal(4, SignalType.DRUM_HIT, 250, -10, 2.5)  # Snare bot
        self.channels[5] = ChannelSignal(5, SignalType.PINK_NOISE, level_db=-15) # HiHat
        
        # Toms
        self.channels[6] = ChannelSignal(6, SignalType.DRUM_HIT, 100, -12, 1.0)
        self.channels[7] = ChannelSignal(7, SignalType.DRUM_HIT, 150, -12, 1.0)
        self.channels[8] = ChannelSignal(8, SignalType.DRUM_HIT, 200, -12, 1.0)
        
        # Overheads
        self.channels[9] = ChannelSignal(9, SignalType.PINK_NOISE, level_db=-20)
        self.channels[10] = ChannelSignal(10, SignalType.PINK_NOISE, level_db=-20)
        
        # Room mics
        self.channels[11] = ChannelSignal(11, SignalType.PINK_NOISE, level_db=-25)
        self.channels[12] = ChannelSignal(12, SignalType.PINK_NOISE, level_db=-25)
        
        # Bass
        self.channels[13] = ChannelSignal(13, SignalType.SINE, 80, -12)   # DI
        self.channels[14] = ChannelSignal(14, SignalType.SINE, 100, -14)  # Amp
        
        # Guitars
        self.channels[15] = ChannelSignal(15, SignalType.SINE, 250, -15)
        self.channels[16] = ChannelSignal(16, SignalType.SINE, 250, -15)
        
        # Keys
        self.channels[17] = ChannelSignal(17, SignalType.SINE, 440, -18)
        self.channels[18] = ChannelSignal(18, SignalType.SINE, 440, -18)
        
        # Vocals
        self.channels[19] = ChannelSignal(19, SignalType.VOCAL_AHH, 440, -12)
        self.channels[20] = ChannelSignal(20, SignalType.VOCAL_AHH, 330, -18)
        self.channels[21] = ChannelSignal(21, SignalType.VOCAL_AHH, 330, -18)
        self.channels[22] = ChannelSignal(22, SignalType.VOCAL_AHH, 330, -20)
        
        # Rest are silence
        for i in range(23, 33):
            self.channels[i] = ChannelSignal(i, SignalType.SILENCE, level_db=-100)
    
    def generate_frame(self, ch: int, time_sec: float, duration: float = 0.01) -> float:
        """
        Generate single frame of audio for channel.
        
        Args:
            ch: Channel number
            time_sec: Current time in seconds
            duration: Frame duration
            
        Returns:
            Signal level in dB
        """
        if ch not in self.channels:
            return -100.0
        
        sig = self.channels[ch]
        
        if sig.signal_type == SignalType.SILENCE:
            return -100.0
        
        elif sig.signal_type == SignalType.SINE:
            # Simple sine wave
            return sig.level_db
        
        elif sig.signal_type == SignalType.PINK_NOISE:
            # Pink noise (constant level with variations)
            variation = np.random.uniform(-3, 3)
            return sig.level_db + variation
        
        elif sig.signal_type == SignalType.DRUM_HIT:
            # Drum hit with rhythm
            if sig.rhythm > 0:
                beat_time = 1.0 / sig.rhythm
                phase = time_sec % beat_time
                
                # Attack phase
                if phase < 0.05:  # 50ms attack
                    return sig.level_db
                # Decay phase
                elif phase < 0.3:  # 300ms decay
                    decay = (0.3 - phase) / 0.25
                    return sig.level_db + 20 * np.log10(decay + 0.01)
                else:
                    return -100.0
            else:
                return -100.0
        
        elif sig.signal_type == SignalType.VOCAL_AHH:
            # Vocal with natural variation
            variation = np.sin(2 * np.pi * 3 * time_sec) * 2  # 3Hz vibrato
            breath = np.random.uniform(-1, 0)  # Breathing
            return sig.level_db + variation + breath
        
        elif sig.signal_type == SignalType.MIXED:
            # Mixed signal
            return sig.level_db + np.random.uniform(-6, 3)
        
        return -100.0
    
    def generate_all(self, time_sec: float) -> Dict[int, float]:
        """Generate signals for all channels at given time."""
        return {ch: self.generate_frame(ch, time_sec) for ch in range(1, 33)}
    
    def set_channel_signal(self, ch: int, signal_type: SignalType, **kwargs):
        """Configure channel signal."""
        if ch in self.channels:
            self.channels[ch].signal_type = signal_type
            for key, value in kwargs.items():
                setattr(self.channels[ch], key, value)
            logger.info(f"CH{ch:02d}: {signal_type.value} configured")
    
    def start_song(self, bpm: float = 120):
        """Configure signals for song playback."""
        beat_duration = 60.0 / bpm
        
        # Kick on beats 1, 3
        self.channels[1].rhythm = 2.0 / beat_duration
        
        # Snare on beats 2, 4
        self.channels[3].rhythm = 2.0 / beat_duration
        self.channels[3].onsets = [beat_duration * 1, beat_duration * 3]
        
        # HiHat on 8th notes
        self.channels[5].signal_type = SignalType.PINK_NOISE
        self.channels[5].level_db = -18
        
        logger.info(f"Song started at {bpm} BPM")


async def demo_generator():
    """Demo of signal generator."""
    logging.basicConfig(level=logging.INFO)
    
    gen = TestSignalGenerator()
    gen.start_song(bpm=120)
    
    print("Generating test signals...")
    print("Time | CH01(Kick) | CH03(Snare) | CH19(Vox)")
    print("-" * 50)
    
    start_time = 0.0
    for i in range(20):
        t = start_time + i * 0.1
        signals = gen.generate_all(t)
        
        kick = signals.get(1, -100)
        snare = signals.get(3, -100)
        vocal = signals.get(19, -100)
        
        print(f"{t:.1f}s | {kick:6.1f}dB | {snare:6.1f}dB | {vocal:6.1f}dB")
        
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(demo_generator())
