"""Apply deterministic, idempotent hardening patches to DSP-only modules.

This helper exists so large legacy modules can receive small auditable edits
without replacing their entire files. It does not touch mixer clients, OSC/MIDI
translators, live write policies, or actuator code.
"""
from pathlib import Path


def replace_once(path: str, old: str, new: str) -> bool:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"expected patch anchor not found in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_true_peak() -> None:
    path = "backend/lufs_gain_staging.py"
    replace_once(
        path,
        "        self._design_interpolation_filter()\n        \n        # Состояние\n",
        "        self._design_interpolation_filter()\n        # FIR interpolation is a streaming filter. Preserve its delay-line\n        # state across host blocks so true-peak does not depend on chunking.\n        self._filter_state = np.zeros(len(self.interp_filter) - 1, dtype=np.float64)\n        \n        # Состояние\n",
    )
    replace_once(
        path,
        "        # Upsample сигнал\n        upsampled = np.zeros(len(samples) * self.oversample_factor)\n        upsampled[::self.oversample_factor] = samples\n",
        "        samples = np.nan_to_num(\n            np.asarray(samples, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0\n        )\n        if samples.size == 0:\n            self._current_peak = 0.0\n            return -100.0\n\n        # Upsample сигнал\n        upsampled = np.zeros(len(samples) * self.oversample_factor, dtype=np.float64)\n        upsampled[::self.oversample_factor] = samples\n",
    )
    replace_once(
        path,
        "        interpolated = signal.lfilter(self.interp_filter, 1, upsampled)\n        interpolated *= self.oversample_factor  # Компенсация амплитуды\n",
        "        interpolated, self._filter_state = signal.lfilter(\n            self.interp_filter, 1, upsampled, zi=self._filter_state\n        )\n        interpolated *= self.oversample_factor  # Компенсация амплитуды\n",
    )
    replace_once(
        path,
        "        self._max_peak = 0.0\n        self._current_peak = 0.0\n\n\nclass AGCEnvelope:",
        "        self._max_peak = 0.0\n        self._current_peak = 0.0\n        self._filter_state = np.zeros(len(self.interp_filter) - 1, dtype=np.float64)\n\n\nclass AGCEnvelope:",
    )


def patch_signal_metrics() -> None:
    path = "backend/signal_metrics.py"
    replace_once(
        path,
        "        samples = np.asarray(samples, dtype=np.float32)\n        self._time_sec += len(samples) / self.sample_rate\n",
        "        samples = np.asarray(samples, dtype=np.float32)\n        # Audio devices, decoders and ML bridges can occasionally surface\n        # non-finite samples. One NaN must never poison level/spectral state.\n        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)\n        self._time_sec += len(samples) / self.sample_rate\n",
    )
    replace_once(
        path,
        "    a = samples_a[:min_len].astype(np.float32)\n    b = samples_b[:min_len].astype(np.float32)\n",
        "    a = np.nan_to_num(samples_a[:min_len].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)\n    b = np.nan_to_num(samples_b[:min_len].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)\n",
    )


def patch_mastering() -> None:
    path = "backend/auto_mastering.py"
    replace_once(
        path,
        "        audio = self._normalize_audio_shape(audio.astype(np.float32))\n",
        "        audio = self._normalize_audio_shape(audio.astype(np.float32))\n        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)\n",
    )
    replace_once(
        path,
        "        # 3. Loudness normalization\n        current_rms = np.sqrt(np.mean(self._monitor_signal(processed) ** 2) + 1e-12)\n        current_db = 20 * np.log10(current_rms)\n        gain_db = self.target_lufs - current_db\n",
        "        # 3. Loudness normalization. The public target is LUFS, so do\n        # not substitute unweighted RMS dB here.\n        current_lufs = self._estimate_lufs(processed)\n        gain_db = self.target_lufs - current_lufs\n",
    )
    replace_once(
        path,
        "        rms_db = float(20 * np.log10(np.sqrt(np.mean(processed ** 2)) + 1e-10))\n\n        return MasteringResult(\n            audio=processed, peak_db=peak_db, lufs=rms_db,\n",
        "        measured_lufs = self._estimate_lufs(processed)\n\n        return MasteringResult(\n            audio=processed, peak_db=peak_db, lufs=measured_lufs,\n",
    )
    replace_once(
        path,
        "    def _estimate_lufs(audio: np.ndarray) -> float:\n        \"\"\"RMS-based LUFS approximation used by compatibility tests.\"\"\"\n        if audio.size == 0:\n            return -100.0\n        if audio.ndim > 1:\n            audio = np.mean(audio, axis=-1)\n        rms = np.sqrt(np.mean(np.square(audio.astype(np.float64))) + 1e-12)\n        return float(20.0 * np.log10(rms + 1e-12))\n",
        "    def _estimate_lufs(self, audio: np.ndarray) -> float:\n        \"\"\"Estimate integrated programme loudness in LUFS.\n\n        Prefer pyloudnorm/BS.1770. For very short buffers or unavailable\n        dependencies, fall back to K-weighted-style RMS offset rather than\n        silently treating plain RMS dB as LUFS.\n        \"\"\"\n        arr = self._normalize_audio_shape(np.asarray(audio, dtype=np.float32))\n        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)\n        if arr.size == 0:\n            return -100.0\n        try:\n            import pyloudnorm as pyln\n            # pyloudnorm requires enough programme for its block size.\n            if len(arr) >= int(0.4 * self.sample_rate):\n                meter = pyln.Meter(self.sample_rate, block_size=0.4)\n                value = float(meter.integrated_loudness(arr))\n                if np.isfinite(value):\n                    return value\n        except Exception:\n            pass\n        mono = self._monitor_signal(arr).astype(np.float64, copy=False)\n        rms = np.sqrt(np.mean(np.square(mono)) + 1e-12)\n        return float(-0.691 + 20.0 * np.log10(rms + 1e-12))\n",
    )
    replace_once(
        path,
        "        peak = np.max(np.abs(audio))\n        reduction_db = 0.0\n\n        if peak > ceiling_lin:\n            gain = ceiling_lin / peak\n",
        "        # Limit against reconstructed 4x peak, not only stored sample\n        # values. This catches inter-sample overs created by reconstruction.\n        try:\n            from scipy.signal import resample_poly\n            reconstructed = resample_poly(audio, 4, 1, axis=0)\n            peak = float(np.max(np.abs(reconstructed)))\n        except Exception:\n            peak = float(np.max(np.abs(audio)))\n        reduction_db = 0.0\n\n        if peak > ceiling_lin:\n            gain = ceiling_lin / peak\n",
    )


if __name__ == "__main__":
    patch_true_peak()
    patch_signal_metrics()
    patch_mastering()
    print("safe DSP hardening applied")
