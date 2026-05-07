"""Offline sandbox renderer for candidate mixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ai_mixing_pipeline.audio_utils import (
    db_to_amp,
    ensure_stereo,
    limit_peak,
    loudness_match_to,
    match_length,
    measure_audio,
    write_audio,
)
from ai_mixing_pipeline.models import CandidateAction, MixCandidate, RenderResult


class SandboxRenderer:
    """Render candidates locally without touching real mixer state."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        safety = self.config.get("safety", {}) or {}
        offline = self.config.get("offline_test", {}) or {}
        self.safety_peak_limit_dbfs = float(safety.get("max_true_peak_dbfs", -1.0))
        self.safe_render_peak_margin_db = max(
            0.0,
            float(offline.get("safe_render_peak_margin_db", 0.6) or 0.0),
        )
        explicit_ceiling = offline.get("safe_render_peak_ceiling_dbfs")
        if explicit_ceiling is None:
            self.peak_ceiling_dbfs = self.safety_peak_limit_dbfs - self.safe_render_peak_margin_db
        else:
            self.peak_ceiling_dbfs = min(float(explicit_ceiling), self.safety_peak_limit_dbfs)

    def render_initial_mix(
        self,
        stems: dict[str, np.ndarray],
        stem_roles: dict[str, str],
        sample_rate: int,
        output_path: str | Path,
    ) -> RenderResult:
        mix, audit = self._sum_processed_stems(stems, stem_roles, [], sample_rate)
        mix, output_gain = limit_peak(mix, self.peak_ceiling_dbfs)
        path = write_audio(output_path, mix, sample_rate)
        metrics = measure_audio(mix, sample_rate)
        return RenderResult(
            candidate_id="000_initial_mix",
            path=str(path),
            sample_rate=sample_rate,
            duration_sec=round(len(mix) / float(max(1, sample_rate)), 3),
            output_gain_db=output_gain,
            audit=audit,
            metrics=metrics,
            metadata=self._render_policy_metadata(),
        )

    def render_candidate(
        self,
        candidate: MixCandidate,
        stems: dict[str, np.ndarray],
        stem_roles: dict[str, str],
        sample_rate: int,
        output_dir: str | Path,
        *,
        target_lufs: float | None = None,
        loudness_match: bool = True,
    ) -> RenderResult:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        if candidate.candidate_id == "000_initial_mix":
            filename = candidate.render_filename or "000_initial_mix.wav"
            return self.render_initial_mix(stems, stem_roles, sample_rate, output_dir / filename)

        mix, audit = self._sum_processed_stems(stems, stem_roles, candidate.actions, sample_rate)
        output_gain = 0.0
        loudness_matched = False
        if loudness_match:
            mix, output_gain = loudness_match_to(
                mix,
                sample_rate,
                target_lufs,
                peak_ceiling_dbfs=self.peak_ceiling_dbfs,
            )
            loudness_matched = target_lufs is not None
        else:
            mix, output_gain = limit_peak(mix, self.peak_ceiling_dbfs)
        filename = candidate.render_filename or f"{candidate.candidate_id}.wav"
        path = write_audio(output_dir / filename, mix, sample_rate)
        metrics = measure_audio(mix, sample_rate)
        return RenderResult(
            candidate_id=candidate.candidate_id,
            path=str(path),
            sample_rate=sample_rate,
            duration_sec=round(len(mix) / float(max(1, sample_rate)), 3),
            loudness_matched=loudness_matched,
            output_gain_db=output_gain,
            audit=audit,
            metrics=metrics,
            metadata=self._render_policy_metadata(),
        )

    def _render_policy_metadata(self) -> dict[str, Any]:
        return {
            "safe_render_policy": "pre_safety_true_peak_trim",
            "safety_peak_limit_dbfs": self.safety_peak_limit_dbfs,
            "render_peak_ceiling_dbfs": self.peak_ceiling_dbfs,
            "safe_render_peak_margin_db": self.safe_render_peak_margin_db,
        }

    def _sum_processed_stems(
        self,
        stems: dict[str, np.ndarray],
        stem_roles: dict[str, str],
        actions: list[CandidateAction],
        sample_rate: int,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not stems:
            raise ValueError("Sandbox rendering requires at least one stem")
        length = max(len(audio) for audio in stems.values())
        mix = np.zeros((length, 2), dtype=np.float32)
        audit: list[dict[str, Any]] = []
        for name, audio in stems.items():
            role = stem_roles.get(name, "unknown")
            processed = self._apply_default_pan(audio, name, role)
            for action in actions:
                if not self._target_matches(action.target, name, role):
                    continue
                before_peak = float(np.max(np.abs(processed)) + 1e-12)
                processed, status = self._apply_action(processed, action, sample_rate)
                after_peak = float(np.max(np.abs(processed)) + 1e-12)
                audit.append(
                    {
                        "candidate_action": action.to_dict(),
                        "stem": name,
                        "role": role,
                        "status": status,
                        "before_peak": before_peak,
                        "after_peak": after_peak,
                    }
                )
            processed = ensure_stereo(processed)
            mix += match_length(processed, length)[:, :2]
        return mix.astype(np.float32), audit

    def _apply_action(
        self,
        audio: np.ndarray,
        action: CandidateAction,
        sample_rate: int,
    ) -> tuple[np.ndarray, str]:
        action_type = action.action_type
        params = action.parameters
        if action_type == "gain_change":
            gain_db = float(params.get("gain_db", 0.0))
            return (audio * db_to_amp(gain_db)).astype(np.float32), "applied"
        if action_type == "high_pass_filter":
            return self._high_pass(audio, sample_rate, float(params.get("frequency_hz", 70.0))), "applied"
        if action_type == "eq_correction":
            return self._peaking_eq(
                audio,
                sample_rate,
                float(params.get("frequency_hz", 1000.0)),
                float(params.get("gain_db", 0.0)),
                float(params.get("q", 1.0)),
            ), "applied"
        if action_type == "compression_correction":
            return self._compress(audio, params), "applied"
        if action_type == "pan_change":
            return self._pan(audio, float(params.get("pan", 0.0))), "applied"
        return audio.astype(np.float32, copy=False), "logged_not_rendered"

    def _high_pass(self, audio: np.ndarray, sample_rate: int, frequency_hz: float) -> np.ndarray:
        try:
            from scipy.signal import butter, sosfilt

            data = np.asarray(audio, dtype=np.float32)
            cutoff = max(10.0, min(frequency_hz, 0.45 * float(sample_rate)))
            sos = butter(2, cutoff, btype="highpass", fs=int(sample_rate), output="sos")
            return sosfilt(sos, data, axis=0).astype(np.float32)
        except Exception:
            return np.asarray(audio, dtype=np.float32)

    def _peaking_eq(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frequency_hz: float,
        gain_db: float,
        q: float,
    ) -> np.ndarray:
        try:
            from mix_agent.actions.base import apply_parametric_eq

            return apply_parametric_eq(audio, int(sample_rate), frequency_hz, gain_db, q)
        except Exception:
            return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _compress(audio: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        data = np.asarray(audio, dtype=np.float32)
        threshold_db = float(params.get("threshold_db", -18.0))
        ratio = max(1.0, min(4.0, float(params.get("ratio", 1.5))))
        makeup_db = max(-3.0, min(3.0, float(params.get("makeup_db", 0.0))))
        abs_data = np.abs(data) + 1e-12
        level_db = 20.0 * np.log10(abs_data)
        over_db = np.maximum(0.0, level_db - threshold_db)
        gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
        gain = np.power(10.0, (-gain_reduction_db + makeup_db) / 20.0)
        return (data * gain).astype(np.float32)

    @staticmethod
    def _pan(audio: np.ndarray, pan: float) -> np.ndarray:
        data = ensure_stereo(audio)
        pan = max(-1.0, min(1.0, float(pan)))
        left_gain = float(np.cos((pan + 1.0) * np.pi / 4.0))
        right_gain = float(np.sin((pan + 1.0) * np.pi / 4.0))
        mono = np.mean(data, axis=1)
        return np.stack([mono * left_gain, mono * right_gain], axis=1).astype(np.float32)

    def _apply_default_pan(self, audio: np.ndarray, stem_name: str, role: str) -> np.ndarray:
        pan = self._default_pan(stem_name, role)
        return self._pan(audio, pan)

    @staticmethod
    def _default_pan(stem_name: str, role: str) -> float:
        label = stem_name.lower().replace("_", " ").replace("-", " ")
        if label.endswith(" l") or " left" in label:
            return -0.65
        if label.endswith(" r") or " right" in label:
            return 0.65
        if role in {"kick", "snare", "bass", "lead_vocal", "vocal"}:
            return 0.0
        return 0.0

    @staticmethod
    def _target_matches(target: str, stem_name: str, role: str) -> bool:
        target = str(target).lower()
        return target in {
            stem_name.lower(),
            role.lower(),
            "mix",
            "all",
        } or (target == "accompaniment" and "vocal" not in role.lower())
