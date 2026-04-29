"""Dependency-light offline virtual mixer fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_mixing_pipeline.audio_utils import (
    audio_files,
    db_to_amp,
    ensure_stereo,
    limit_peak,
    match_length,
    measure_audio,
    read_audio,
    write_audio,
)

from .action_schema import CandidateActionSet
from .ayaic_balance import compute_input_gain_db
from .mixer_state import ChannelState, MixerState, role_from_channel_map
from .virtual_mixer_base import VirtualMixer


class FallbackVirtualMixer(VirtualMixer):
    """Small offline summing mixer with gain, pan, and clipping protection."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.sample_rate = int(self.config.get("sample_rate", 48000))
        self.prevent_clipping = bool(self.config.get("prevent_clipping", True))
        self.peak_ceiling_dbfs = float((self.config.get("safety", {}) or {}).get("max_true_peak_dbfs", -1.0))
        self.state = MixerState(sample_rate=self.sample_rate, metadata={"virtual_mixer": "fallback_virtual_mixer"})
        self._audio: dict[str, np.ndarray] = {}
        self.warnings: list[str] = []

    def load_project(self, multitrack_dir: str | Path, channel_map: dict[str, Any] | None = None) -> dict[str, Any]:
        channel_map = dict(channel_map or {})
        root = Path(multitrack_dir).expanduser()
        files = audio_files(root)
        if not files:
            raise ValueError(f"No supported audio files found in {root}")
        self._audio.clear()
        self.state.channels.clear()
        balance_reports: list[dict[str, Any]] = []
        ayaic_config = dict(self.config.get("ayaic_balance", {}) or {})
        for path in files:
            audio, sample_rate = read_audio(path, target_sample_rate=self.sample_rate)
            self.sample_rate = sample_rate
            channel_id = path.stem
            self._audio[channel_id] = audio
            role = role_from_channel_map(path, channel_map)
            if bool(ayaic_config.get("enabled", True)):
                gain_db, balance_report = compute_input_gain_db(channel_id, role, audio, ayaic_config)
            else:
                gain_db = self._default_gain_db(path.stem, role)
                balance_report = {
                    "method": "static_role_default",
                    "channel": channel_id,
                    "role": role,
                    "applied_gain_db": gain_db,
                }
            balance_reports.append(balance_report)
            self.state.channels[channel_id] = ChannelState(
                channel_id=channel_id,
                path=str(path),
                role=role,
                gain_db=gain_db,
                pan=self._default_pan(path.stem),
            )
        self.state.sample_rate = self.sample_rate
        self.state.metadata["initial_gain_balance"] = "ayaic_level_plane_input" if bool(ayaic_config.get("enabled", True)) else "role_based_static_rough_mix"
        self.state.metadata["initial_gain_balance_report"] = balance_reports
        return self.export_state()

    def render(self, actions: CandidateActionSet, output_path: str | Path) -> dict[str, Any]:
        if not self._audio:
            raise RuntimeError("load_project must be called before render")
        warnings: list[str] = []
        length = max(len(audio) for audio in self._audio.values())
        mix = np.zeros((length, 2), dtype=np.float32)
        action_audit: list[dict[str, Any]] = []
        action_by_channel: dict[str, list[Any]] = {}
        master_gain_db = 0.0
        master_actions: list[Any] = []
        for action in actions.actions:
            channel_id = getattr(action, "channel_id", "mix")
            if channel_id == "master":
                if action.action_type == "gain":
                    master_gain_db += float(getattr(action, "gain_db", 0.0))
                    action_audit.append({"action": action.to_dict(), "status": "applied_master"})
                else:
                    master_actions.append(action)
                continue
            action_by_channel.setdefault(channel_id, []).append(action)

        for channel_id, audio in self._audio.items():
            channel_state = self.state.channels[channel_id]
            processed = ensure_stereo(audio)
            gain_db = channel_state.gain_db
            pan = channel_state.pan
            for action in action_by_channel.get(channel_id, []):
                if action.action_type == "gain":
                    gain_db += float(getattr(action, "gain_db", 0.0))
                    action_audit.append({"action": action.to_dict(), "status": "applied"})
                elif action.action_type == "pan":
                    pan = float(getattr(action, "pan", pan))
                    action_audit.append({"action": action.to_dict(), "status": "applied"})
                elif action.action_type == "eq":
                    processed = self._apply_eq_action(processed, action)
                    action_audit.append({"action": action.to_dict(), "status": "applied"})
                elif action.action_type == "compressor":
                    processed = self._apply_compressor_action(processed, action)
                    action_audit.append({"action": action.to_dict(), "status": "applied"})
                elif action.action_type == "gate_expander":
                    processed = self._apply_gate_expander_action(processed, action)
                    action_audit.append({"action": action.to_dict(), "status": "applied"})
                elif action.action_type == "no_change":
                    action_audit.append({"action": action.to_dict(), "status": "no_change"})
                else:
                    warnings.append(f"{action.action_type} unsupported by fallback_virtual_mixer; logged but skipped.")
                    action_audit.append({"action": action.to_dict(), "status": "skipped_unsupported"})
            processed = self._apply_pan(processed * db_to_amp(gain_db), pan)
            mix += match_length(processed, length)[:, :2]
        for action in master_actions:
            if action.action_type == "eq":
                mix = self._apply_eq_action(mix, action)
                action_audit.append({"action": action.to_dict(), "status": "applied_master"})
            elif action.action_type == "compressor":
                mix = self._apply_compressor_action(mix, action)
                action_audit.append({"action": action.to_dict(), "status": "applied_master"})
            else:
                warnings.append(f"{action.action_type} unsupported on master by fallback_virtual_mixer; logged but skipped.")
                action_audit.append({"action": action.to_dict(), "status": "skipped_unsupported"})
        if master_gain_db:
            mix = mix * db_to_amp(master_gain_db)
        output_gain_db = 0.0
        if self.prevent_clipping:
            mix, output_gain_db = limit_peak(mix, self.peak_ceiling_dbfs)
        path = write_audio(output_path, mix, self.sample_rate)
        metrics = measure_audio(mix, self.sample_rate)
        return {
            "candidate_id": actions.candidate_id,
            "path": str(path),
            "sample_rate": self.sample_rate,
            "duration_sec": round(len(mix) / float(max(1, self.sample_rate)), 3),
            "output_gain_db": output_gain_db,
            "warnings": warnings,
            "audit": action_audit,
            "metrics": metrics,
            "virtual_mixer": "fallback_virtual_mixer",
            "osc_midi_sent": False,
        }

    def export_state(self) -> dict[str, Any]:
        return self.state.to_dict()

    def import_state(self, state: dict[str, Any]) -> None:
        self.state.metadata.update(dict(state.get("metadata", {}) or {}))

    @staticmethod
    def _apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
        data = ensure_stereo(audio)
        pan = max(-1.0, min(1.0, float(pan)))
        left_gain = float(np.cos((pan + 1.0) * np.pi / 4.0))
        right_gain = float(np.sin((pan + 1.0) * np.pi / 4.0))
        mono = np.mean(data, axis=1)
        return np.stack([mono * left_gain, mono * right_gain], axis=1).astype(np.float32)

    def _apply_eq_action(self, audio: np.ndarray, action: Any) -> np.ndarray:
        freq_hz = float(getattr(action, "freq_hz", 1000.0))
        gain_db = float(getattr(action, "gain_db", 0.0))
        q = max(0.1, float(getattr(action, "q", 1.0)))
        filter_type = str(getattr(action, "filter_type", "peaking"))
        if abs(gain_db) < 1e-6:
            return np.asarray(audio, dtype=np.float32)
        if filter_type in {"high_pass", "hpf"}:
            return self._high_pass(audio, freq_hz)
        return self._peaking_eq(audio, freq_hz, gain_db, q)

    def _high_pass(self, audio: np.ndarray, freq_hz: float) -> np.ndarray:
        try:
            from scipy.signal import butter, sosfilt

            cutoff = max(10.0, min(float(freq_hz), 0.45 * float(self.sample_rate)))
            sos = butter(2, cutoff, btype="highpass", fs=int(self.sample_rate), output="sos")
            return sosfilt(sos, ensure_stereo(audio), axis=0).astype(np.float32)
        except Exception:
            return np.asarray(audio, dtype=np.float32)

    def _peaking_eq(self, audio: np.ndarray, freq_hz: float, gain_db: float, q: float) -> np.ndarray:
        data = ensure_stereo(audio)
        freq_hz = max(20.0, min(float(freq_hz), 0.45 * float(self.sample_rate)))
        a = 10.0 ** (float(gain_db) / 40.0)
        omega = 2.0 * np.pi * freq_hz / float(self.sample_rate)
        alpha = np.sin(omega) / (2.0 * q)
        cos_w = np.cos(omega)
        b0 = 1.0 + alpha * a
        b1 = -2.0 * cos_w
        b2 = 1.0 - alpha * a
        a0 = 1.0 + alpha / a
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha / a
        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        acoef = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
        try:
            from scipy.signal import lfilter

            return lfilter(b, acoef, data, axis=0).astype(np.float32)
        except Exception:
            out = np.zeros_like(data, dtype=np.float32)
            for channel in range(data.shape[1]):
                x1 = x2 = y1 = y2 = 0.0
                for index, x0 in enumerate(data[:, channel]):
                    y0 = b[0] * x0 + b[1] * x1 + b[2] * x2 - acoef[1] * y1 - acoef[2] * y2
                    out[index, channel] = y0
                    x2, x1 = x1, float(x0)
                    y2, y1 = y1, float(y0)
            return out

    def _apply_compressor_action(self, audio: np.ndarray, action: Any) -> np.ndarray:
        data = np.asarray(audio, dtype=np.float32)
        threshold_db = float(getattr(action, "threshold_db", -18.0))
        ratio = max(1.0, min(10.0, float(getattr(action, "ratio", 1.5))))
        attack_ms = max(0.1, float(getattr(action, "attack_ms", 15.0)))
        release_ms = max(1.0, float(getattr(action, "release_ms", 140.0)))
        makeup_db = max(-12.0, min(12.0, float(getattr(action, "makeup_gain_db", 0.0))))
        envelope = self._smoothed_envelope_db(data, attack_ms, release_ms)
        over_db = np.maximum(0.0, envelope - threshold_db)
        gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
        gain = np.power(10.0, (-gain_reduction_db + makeup_db) / 20.0)
        if data.ndim == 2 and gain.ndim == 1:
            gain = gain[:, None]
        return (data * gain).astype(np.float32)

    def _apply_gate_expander_action(self, audio: np.ndarray, action: Any) -> np.ndarray:
        data = np.asarray(audio, dtype=np.float32)
        threshold_db = float(getattr(action, "threshold_db", -50.0))
        ratio = max(1.0, min(8.0, float(getattr(action, "ratio", 2.0))))
        attack_ms = max(0.1, float(getattr(action, "attack_ms", 5.0)))
        release_ms = max(1.0, float(getattr(action, "release_ms", 120.0)))
        envelope = self._smoothed_envelope_db(data, attack_ms, release_ms)
        below_db = np.maximum(0.0, threshold_db - envelope)
        attenuation_db = below_db * (1.0 - 1.0 / ratio)
        gain = np.power(10.0, -attenuation_db / 20.0)
        if data.ndim == 2 and gain.ndim == 1:
            gain = gain[:, None]
        return (data * gain).astype(np.float32)

    def _smoothed_envelope_db(self, audio: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
        mono = np.mean(np.abs(ensure_stereo(audio)), axis=1)
        attack_coeff = np.exp(-1.0 / max(1.0, self.sample_rate * attack_ms / 1000.0))
        release_coeff = np.exp(-1.0 / max(1.0, self.sample_rate * release_ms / 1000.0))
        envelope = np.zeros_like(mono, dtype=np.float32)
        current = 0.0
        for index, value in enumerate(mono):
            coeff = attack_coeff if value > current else release_coeff
            current = float(coeff * current + (1.0 - coeff) * value)
            envelope[index] = current
        return 20.0 * np.log10(np.maximum(envelope, 1e-9))

    @staticmethod
    def _default_pan(name: str) -> float:
        label = name.lower().replace("_", " ").replace("-", " ")
        if label.endswith(" l") or " left" in label:
            return -0.65
        if label.endswith(" r") or " right" in label:
            return 0.65
        return 0.0

    @staticmethod
    def _default_gain_db(name: str, role: str) -> float:
        label = name.lower().replace("_", " ").replace("-", " ")
        role = role.lower()
        if "back vox" in label or "backing" in label:
            return -11.0
        if "playback" in label:
            return -13.0
        if "oh " in label or "overhead" in label:
            return -12.0
        if "tom" in label:
            return -11.0
        if "kick" in role:
            return -7.0
        if "snare" in role:
            return -9.0
        if "bass" in role:
            return -7.5
        if "vocal" in role:
            return -5.0
        if "guitar" in role:
            return -11.0
        if role in {"keys", "piano", "synth"}:
            return -12.0
        if "drum" in role:
            return -11.0
        return -14.0
