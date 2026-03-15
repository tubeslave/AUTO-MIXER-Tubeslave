"""
Reference-based automatic mastering using Matchering.

Falls back to simple loudness matching + limiting when Matchering is unavailable.
"""

import logging
import os
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matchering as mg
    HAS_MATCHERING = True
except ImportError:
    HAS_MATCHERING = False

try:
    from scipy import signal as scipy_signal
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AutoMaster:
    """Automatic mastering with reference matching."""

    def __init__(self, target_lufs: float = -14.0,
                 true_peak_limit: float = -1.0,
                 sample_rate: int = 48000):
        self.target_lufs = target_lufs
        self.true_peak_limit = true_peak_limit
        self.sample_rate = sample_rate

    def master(self, input_audio: np.ndarray,
               reference_audio: np.ndarray,
               sr: int = 48000) -> np.ndarray:
        """Master input audio to match the reference.

        Args:
            input_audio: Input audio array (mono or stereo).
            reference_audio: Reference audio array.
            sr: Sample rate.

        Returns:
            Mastered audio as numpy array.
        """
        if HAS_MATCHERING:
            return self._master_matchering(input_audio, reference_audio, sr)
        return self._master_fallback(input_audio, reference_audio, sr)

    def master_file(self, input_path: str, reference_path: str,
                    output_path: str) -> bool:
        """Master a file to match a reference file.

        Args:
            input_path: Path to input WAV file.
            reference_path: Path to reference WAV file.
            output_path: Path for output mastered WAV file.

        Returns:
            True if mastering was successful.
        """
        if HAS_MATCHERING:
            try:
                mg.process(
                    target=input_path,
                    reference=reference_path,
                    results=[mg.pcm16(output_path)],
                )
                logger.info(f"Mastered file saved: {output_path}")
                return True
            except Exception as e:
                logger.error(f"Matchering failed: {e}")
                return self._master_file_fallback(input_path, reference_path, output_path)

        return self._master_file_fallback(input_path, reference_path, output_path)

    def _master_matchering(self, input_audio: np.ndarray,
                           reference_audio: np.ndarray,
                           sr: int) -> np.ndarray:
        """Use Matchering library for mastering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "input.wav")
            ref_path = os.path.join(tmpdir, "reference.wav")
            out_path = os.path.join(tmpdir, "output.wav")

            self._write_wav(in_path, input_audio, sr)
            self._write_wav(ref_path, reference_audio, sr)

            try:
                mg.process(
                    target=in_path,
                    reference=ref_path,
                    results=[mg.pcm16(out_path)],
                )
                return self._read_wav(out_path)
            except Exception as e:
                logger.error(f"Matchering failed, using fallback: {e}")
                return self._master_fallback(input_audio, reference_audio, sr)

    def _master_fallback(self, input_audio: np.ndarray,
                         reference_audio: np.ndarray,
                         sr: int) -> np.ndarray:
        """Simple loudness matching + limiting fallback."""
        input_lufs = self._estimate_lufs(input_audio)
        ref_lufs = self._estimate_lufs(reference_audio)

        if ref_lufs > -70 and input_lufs > -70:
            gain_db = ref_lufs - input_lufs
        else:
            gain_db = self.target_lufs - input_lufs if input_lufs > -70 else 0.0

        gain_db = max(min(gain_db, 12.0), -12.0)
        gain_linear = 10.0 ** (gain_db / 20.0)

        result = input_audio.astype(np.float64) * gain_linear

        result = self._apply_eq_match(result, reference_audio, sr)
        result = self._limit(result)

        return result.astype(np.float32)

    def _apply_eq_match(self, audio: np.ndarray,
                        reference: np.ndarray, sr: int) -> np.ndarray:
        """Simple spectral matching via frequency-domain transfer function."""
        if not HAS_SCIPY:
            return audio

        n = min(len(audio), len(reference))
        fft_size = 2 ** int(np.ceil(np.log2(n)))

        audio_fft = np.fft.rfft(audio[:n], fft_size)
        ref_fft = np.fft.rfft(reference[:n], fft_size)

        audio_mag = np.abs(audio_fft) + 1e-10
        ref_mag = np.abs(ref_fft) + 1e-10

        ratio = ref_mag / audio_mag
        ratio = np.clip(ratio, 0.1, 10.0)

        kernel_size = max(fft_size // 128, 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(ratio, np.ones(kernel_size) / kernel_size, mode='same')

        result_fft = audio_fft * smoothed
        result = np.fft.irfft(result_fft, fft_size)[:n]

        return result

    def _limit(self, audio: np.ndarray) -> np.ndarray:
        """Simple brick-wall limiter."""
        peak_linear = 10.0 ** (self.true_peak_limit / 20.0)
        peak = np.max(np.abs(audio))
        if peak > peak_linear:
            audio = audio * (peak_linear / peak)
        return audio

    @staticmethod
    def _estimate_lufs(audio: np.ndarray) -> float:
        """Estimate integrated LUFS (simplified)."""
        audio = audio.astype(np.float64)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -100.0
        return float(20.0 * np.log10(rms + 1e-10))

    @staticmethod
    def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
        """Write audio to WAV file."""
        if HAS_SCIPY:
            audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            wavfile.write(path, sr, audio_int)
        else:
            import struct
            audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            n_samples = len(audio_int)
            with open(path, "wb") as f:
                f.write(b"RIFF")
                f.write(struct.pack("<I", 36 + n_samples * 2))
                f.write(b"WAVE")
                f.write(b"fmt ")
                f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
                f.write(b"data")
                f.write(struct.pack("<I", n_samples * 2))
                f.write(audio_int.tobytes())

    @staticmethod
    def _read_wav(path: str) -> np.ndarray:
        """Read audio from WAV file."""
        if HAS_SCIPY:
            sr, data = wavfile.read(path)
            return data.astype(np.float32) / 32768.0
        return np.zeros(1, dtype=np.float32)

    def _master_file_fallback(self, input_path: str, reference_path: str,
                              output_path: str) -> bool:
        """Fallback file mastering without Matchering."""
        try:
            input_audio = self._read_wav(input_path)
            ref_audio = self._read_wav(reference_path)
            result = self._master_fallback(input_audio, ref_audio, self.sample_rate)
            self._write_wav(output_path, result, self.sample_rate)
            logger.info(f"Fallback mastering saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Fallback mastering failed: {e}")
            return False
