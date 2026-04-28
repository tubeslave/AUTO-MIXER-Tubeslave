"""Sandbox renderer for decision-layer candidate action sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from ai_mixing_pipeline.audio_utils import loudness_match_to, read_audio, safe_slug, write_audio

from .action_schema import CandidateActionSet, ensure_no_change_candidate
from .virtual_mixer_base import VirtualMixer


class DecisionSandboxRenderer:
    """Render candidate mixes into one run directory and write a manifest."""

    def __init__(self, virtual_mixer: VirtualMixer, output_dir: str | Path, config: dict[str, Any] | None = None):
        self.virtual_mixer = virtual_mixer
        self.output_dir = Path(output_dir).expanduser()
        self.renders_dir = self.output_dir / "renders"
        self.config = dict(config or {})
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.manifest: list[dict[str, Any]] = []

    def render_candidate(self, candidate: CandidateActionSet) -> dict[str, Any]:
        filename = f"{safe_slug(candidate.candidate_id)}.wav"
        result = self.virtual_mixer.render(candidate, self.renders_dir / filename)
        result["candidate"] = candidate.to_dict()
        self.manifest.append(result)
        return result

    def render_candidates(self, candidates: list[CandidateActionSet]) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for candidate in ensure_no_change_candidate(candidates):
            results[candidate.candidate_id] = self.render_candidate(candidate)
        if bool(self.config.get("loudness_match", True)):
            self.loudness_match_candidates(results)
        self.save_render_manifest(results)
        return results

    def loudness_match_candidates(self, results: dict[str, dict[str, Any]]) -> None:
        if not results:
            return
        first = next(iter(results.values()))
        target_lufs = self.config.get("target_lufs")
        if target_lufs is None:
            target_lufs = (first.get("metrics", {}).get("level", {}) or {}).get("integrated_lufs")
        peak_ceiling = float((self.config.get("safety", {}) or {}).get("max_true_peak_dbfs", -1.0))
        for result in results.values():
            audio, sample_rate = read_audio(result["path"])
            matched, output_gain = loudness_match_to(audio, sample_rate, target_lufs, peak_ceiling)
            write_audio(result["path"], matched, sample_rate)
            result["loudness_matched"] = target_lufs is not None
            result["loudness_match_gain_db"] = output_gain

    def save_render_manifest(self, results: dict[str, dict[str, Any]] | None = None) -> Path:
        target = self.output_dir / "reports" / "candidate_manifest.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = list((results or {}).values()) if results is not None else self.manifest
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
        return target
