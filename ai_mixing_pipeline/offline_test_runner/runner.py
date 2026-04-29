"""End-to-end offline AI mixing test chain."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from ai_mixing_pipeline.action_planner import OfflineActionPlanner
from ai_mixing_pipeline.audio_utils import audio_files, measure_audio_file, read_audio, write_audio
from ai_mixing_pipeline.config import load_roles_config
from ai_mixing_pipeline.critics import (
    AudioboxAestheticsCritic,
    CLAPSemanticCritic,
    MuQEvalCritic,
)
from ai_mixing_pipeline.models import AudioTrack, DecisionResult, MixCandidate, RenderResult, jsonable
from ai_mixing_pipeline.reports import (
    write_accepted_rejected_actions,
    write_critic_scores_csv,
    write_decision_log,
    write_json,
    write_summary_report,
)
from ai_mixing_pipeline.safety_governor import SafetyGovernor
from ai_mixing_pipeline.sandbox_renderer import SandboxRenderer
from ai_mixing_pipeline.source_separation import OfflineSourceSeparator
from ai_mixing_pipeline.stem_critics import MERTStemCritic
from ai_mixing_pipeline.technical_analyzers import EssentiaTechnicalAnalyzer, IdentityBleedCritic
from ai_mixing_pipeline.decision_engine import DecisionEngine


@dataclass
class LoadedMultitrack:
    stems: dict[str, np.ndarray]
    stem_roles: dict[str, str]
    tracks: list[AudioTrack]
    sample_rate: int
    channel_map: dict[str, Any]


class OfflineTestRunner:
    """Run load -> analyze -> candidates -> render -> critics -> safety -> best_mix."""

    VALID_MODES = {"observe", "suggest", "offline_test", "shadow_mix", "assisted_offline"}

    def __init__(
        self,
        *,
        input_dir: str | Path = "offline_test_input",
        output_dir: str | Path = "offline_test_output",
        config_path: str | Path = "configs/ai_mixing_roles.yaml",
        mode: str = "offline_test",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported offline AI mixing mode: {mode}")
        self.input_dir = Path(input_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.config_path = Path(config_path).expanduser()
        self.mode = mode
        self.config = load_roles_config(self.config_path)
        self.renders_dir = self.output_dir / "renders"
        self.reports_dir = self.output_dir / "reports"
        self.snapshots_dir = self.output_dir / "snapshots"

    def run(self) -> dict[str, Any]:
        self._prepare_output_dirs()
        loaded = self._load_multitrack()
        reference_path = self._find_reference()
        before_snapshot = self._snapshot("before", loaded, selected_candidate_id="000_initial_mix")
        write_json(self.snapshots_dir / "mixer_state_before.json", before_snapshot)

        renderer = SandboxRenderer(self.config)
        initial_result = renderer.render_initial_mix(
            loaded.stems,
            loaded.stem_roles,
            loaded.sample_rate,
            self.renders_dir / "000_initial_mix.wav",
        )
        initial_metrics = initial_result.metrics
        reference_profile = self._analyze_reference(reference_path)

        planner = OfflineActionPlanner(self.config)
        planner_status = planner.participation_status()
        candidates = planner.generate_candidates(
            stems=loaded.stems,
            stem_roles=loaded.stem_roles,
            initial_metrics=initial_metrics,
            reference_profile=reference_profile,
        )
        if self.mode == "observe":
            candidates = [candidates[0]]

        render_results = self._render_candidates(
            renderer,
            candidates,
            loaded,
            initial_result,
        )
        render_paths = {
            candidate_id: result.path
            for candidate_id, result in render_results.items()
        }
        critics = self._build_critics()
        evaluations = self._evaluate_candidates(candidates, critics, render_paths, initial_result.path)

        governor = SafetyGovernor(self.config)
        safety_results = {
            candidate.candidate_id: governor.evaluate(
                candidate,
                render_paths[candidate.candidate_id],
                critic_results=evaluations.get(candidate.candidate_id, {}),
                enforce_min_improvement=False,
            )
            for candidate in candidates
        }
        decision = DecisionEngine(self.config).choose_best(candidates, evaluations, safety_results)
        decision = self._enforce_final_safety(decision, candidates, governor, render_paths, evaluations, safety_results)

        best_mix_path = ""
        if self.mode in {"offline_test", "shadow_mix", "assisted_offline"}:
            best_mix_path = str(self.renders_dir / "005_best_mix.wav")
            self._copy_best_mix(render_paths[decision.selected_candidate_id], best_mix_path)
            render_paths["005_best_mix"] = best_mix_path

        after_snapshot = self._snapshot(
            "after",
            loaded,
            selected_candidate_id=decision.selected_candidate_id,
            accepted_actions=self._selected_actions(candidates, decision),
            best_mix_path=best_mix_path,
        )
        if self.mode == "shadow_mix":
            after_snapshot["state_applied"] = False
        write_json(self.snapshots_dir / "mixer_state_after.json", after_snapshot)

        source_separation = reference_profile.get("source_separation") if isinstance(reference_profile, dict) else None
        module_status = self._module_status(
            evaluations,
            source_separation=source_separation,
            action_planner=planner_status,
        )
        if self.config.get("offline_test", {}).get("save_reports", True):
            write_decision_log(
                self.reports_dir / "decision_log.jsonl",
                mode=self.mode,
                candidates=candidates,
                evaluations=evaluations,
                decision=decision,
                safety_results=safety_results,
                render_paths=render_paths,
            )
            write_critic_scores_csv(self.reports_dir / "critic_scores.csv", evaluations, decision)
            write_accepted_rejected_actions(self.reports_dir, candidates, decision, safety_results)
            write_summary_report(
                self.reports_dir / "summary_report.md",
                mode=self.mode,
                candidates=candidates,
                evaluations=evaluations,
                decision=decision,
                safety_results=safety_results,
                render_paths=render_paths,
                module_status=module_status,
                source_separation=source_separation,
            )

        return jsonable(
            {
                "mode": self.mode,
                "selected_candidate_id": decision.selected_candidate_id,
                "best_mix_path": best_mix_path,
                "output_dir": str(self.output_dir),
                "renders": render_paths,
                "render_results": {
                    candidate_id: result.to_dict()
                    for candidate_id, result in render_results.items()
                },
                "reports": {
                    "decision_log": str(self.reports_dir / "decision_log.jsonl"),
                    "summary_report": str(self.reports_dir / "summary_report.md"),
                    "critic_scores": str(self.reports_dir / "critic_scores.csv"),
                    "accepted_actions": str(self.reports_dir / "accepted_actions.json"),
                    "rejected_actions": str(self.reports_dir / "rejected_actions.json"),
                },
                "snapshots": {
                    "before": str(self.snapshots_dir / "mixer_state_before.json"),
                    "after": str(self.snapshots_dir / "mixer_state_after.json"),
                },
                "decision": decision.to_dict(),
                "module_status": module_status,
                "source_separation": source_separation,
                "action_planner": planner_status,
            }
        )

    def _prepare_output_dirs(self) -> None:
        self.renders_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _load_multitrack(self) -> LoadedMultitrack:
        multitrack_dir = self.input_dir / "multitrack"
        files = audio_files(multitrack_dir)
        if not files:
            raise ValueError(f"No supported audio files found in {multitrack_dir}")
        channel_map = self._load_channel_map()
        configured_sr = int((self.config.get("offline_test", {}) or {}).get("sample_rate", 0) or 0)
        if configured_sr <= 0:
            _, configured_sr = read_audio(files[0])
        stems: dict[str, np.ndarray] = {}
        stem_roles: dict[str, str] = {}
        tracks: list[AudioTrack] = []
        for path in files:
            audio, sample_rate = read_audio(path, target_sample_rate=configured_sr)
            name = path.stem
            role = self._role_for_path(path, channel_map)
            stems[name] = audio
            stem_roles[name] = role
            tracks.append(
                AudioTrack(
                    name=name,
                    path=str(path),
                    role=role,
                    sample_rate=sample_rate,
                    duration_sec=round(len(audio) / float(max(1, sample_rate)), 3),
                    channels=audio.shape[1] if audio.ndim == 2 else 1,
                )
            )
        return LoadedMultitrack(
            stems=stems,
            stem_roles=stem_roles,
            tracks=tracks,
            sample_rate=configured_sr,
            channel_map=channel_map,
        )

    def _find_reference(self) -> Path | None:
        reference_dir = self.input_dir / "reference"
        preferred = reference_dir / "reference_mix.wav"
        if preferred.exists():
            return preferred
        files = audio_files(reference_dir)
        return files[0] if files else None

    def _load_channel_map(self) -> dict[str, Any]:
        path = self.input_dir / "config" / "channel_map.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}

    def _role_for_path(self, path: Path, channel_map: dict[str, Any]) -> str:
        for key in (path.name, path.stem):
            item = channel_map.get(key)
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                for role_key in ("role", "source_role", "stem_role", "instrument"):
                    if item.get(role_key):
                        return str(item[role_key])
        channels = channel_map.get("channels")
        if isinstance(channels, list):
            for item in channels:
                if not isinstance(item, dict):
                    continue
                if item.get("file") in {path.name, path.stem} or item.get("name") == path.stem:
                    return str(item.get("role") or item.get("source_role") or item.get("instrument") or "unknown")
        try:
            from mix_agent.analysis.loader import infer_stem_role

            return infer_stem_role(path.stem)
        except Exception:
            return "unknown"

    def _render_candidates(
        self,
        renderer: SandboxRenderer,
        candidates: list[MixCandidate],
        loaded: LoadedMultitrack,
        initial_result: RenderResult,
    ) -> dict[str, RenderResult]:
        target_lufs = None
        try:
            target_lufs = float(initial_result.metrics.get("level", {}).get("integrated_lufs"))
        except (TypeError, ValueError):
            target_lufs = None
        loudness_match = bool(self.config.get("offline_test", {}).get("loudness_match_candidates", True))
        results: dict[str, RenderResult] = {}
        for candidate in candidates:
            results[candidate.candidate_id] = renderer.render_candidate(
                candidate,
                loaded.stems,
                loaded.stem_roles,
                loaded.sample_rate,
                self.renders_dir,
                target_lufs=target_lufs,
                loudness_match=loudness_match,
            )
        return results

    def _build_critics(self) -> list[Any]:
        critics_config = self.config.get("critics", {}) or {}
        critics = [
            MuQEvalCritic(critics_config.get("muq_eval", {})),
            AudioboxAestheticsCritic(critics_config.get("audiobox_aesthetics", {})),
            MERTStemCritic(critics_config.get("mert", {})),
            CLAPSemanticCritic(critics_config.get("clap", {})),
            EssentiaTechnicalAnalyzer(critics_config.get("essentia", {})),
            IdentityBleedCritic(critics_config.get("panns_or_beats", {})),
        ]
        return [critic for critic in critics if bool(getattr(critic, "config", {}).get("enabled", True))]

    def _evaluate_candidates(
        self,
        candidates: list[MixCandidate],
        critics: list[Any],
        render_paths: dict[str, str],
        initial_path: str,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        evaluations: dict[str, dict[str, dict[str, Any]]] = {}
        for candidate in candidates:
            by_critic: dict[str, dict[str, Any]] = {}
            after_path = render_paths[candidate.candidate_id]
            for critic in critics:
                try:
                    by_critic[critic.name] = critic.compare(
                        initial_path,
                        after_path,
                        context={
                            "mode": self.mode,
                            "candidate_id": candidate.candidate_id,
                            "embedding_dir": str(self.reports_dir / "embeddings"),
                        },
                    )
                except Exception as exc:
                    by_critic[critic.name] = critic.unavailable_result(str(exc))
            evaluations[candidate.candidate_id] = by_critic
        return evaluations

    def _analyze_reference(self, reference_path: Path | None) -> dict[str, Any]:
        critics_config = self.config.get("critics", {}) or {}
        separator = OfflineSourceSeparator(critics_config.get("demucs_or_openunmix", {}))
        if reference_path is None:
            return {
                "available": False,
                "warnings": ["No reference mix supplied."],
                "analysis": {},
                "source_separation": separator.separate_reference(
                    None,
                    self.output_dir / "reference_separation",
                ),
            }
        profile: dict[str, Any] = {"available": True, "path": str(reference_path), "analysis": {}}
        for critic in (
            EssentiaTechnicalAnalyzer(critics_config.get("essentia", {})),
            MERTStemCritic(critics_config.get("mert", {})),
            CLAPSemanticCritic(critics_config.get("clap", {})),
        ):
            try:
                profile["analysis"][critic.name] = critic.analyze(
                    str(reference_path),
                    context={"embedding_dir": str(self.reports_dir / "reference_embeddings")},
                )
            except Exception as exc:
                profile["analysis"][critic.name] = critic.unavailable_result(str(exc))
        profile["source_separation"] = separator.separate_reference(
            reference_path,
            self.output_dir / "reference_separation",
        )
        return profile

    def _enforce_final_safety(
        self,
        decision: DecisionResult,
        candidates: list[MixCandidate],
        governor: SafetyGovernor,
        render_paths: dict[str, str],
        evaluations: dict[str, dict[str, dict[str, Any]]],
        safety_results: dict[str, Any],
    ) -> DecisionResult:
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        selected = by_id[decision.selected_candidate_id]
        no_change_id = candidates[0].candidate_id
        improvement = decision.final_scores.get(selected.candidate_id, 0.0) - decision.final_scores.get(no_change_id, 0.0)
        final_safety = governor.evaluate(
            selected,
            render_paths[selected.candidate_id],
            critic_results=evaluations.get(selected.candidate_id, {}),
            score_improvement=improvement,
            enforce_min_improvement=selected.candidate_id != no_change_id,
        )
        safety_results[selected.candidate_id] = final_safety
        if not final_safety.passed:
            decision.selected_candidate_id = no_change_id
            decision.no_change_selected = True
            decision.explanations[no_change_id] = (
                decision.explanations.get(no_change_id, "")
                + f"; final Safety Governor fallback from {selected.candidate_id}: {', '.join(final_safety.reasons)}"
            )
        return decision

    @staticmethod
    def _copy_best_mix(source_path: str, best_mix_path: str) -> None:
        target = Path(best_mix_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(source_path, target)
        except Exception:
            audio, sample_rate = read_audio(source_path)
            write_audio(target, audio, sample_rate)

    @staticmethod
    def _selected_actions(candidates: list[MixCandidate], decision: DecisionResult) -> list[dict[str, Any]]:
        for candidate in candidates:
            if candidate.candidate_id == decision.selected_candidate_id and not decision.no_change_selected:
                return [action.to_dict() for action in candidate.actions]
        return []

    def _snapshot(
        self,
        phase: str,
        loaded: LoadedMultitrack,
        *,
        selected_candidate_id: str,
        accepted_actions: list[dict[str, Any]] | None = None,
        best_mix_path: str = "",
    ) -> dict[str, Any]:
        return {
            "phase": phase,
            "mode": self.mode,
            "sample_rate": loaded.sample_rate,
            "selected_candidate_id": selected_candidate_id,
            "accepted_actions": accepted_actions or [],
            "best_mix_path": best_mix_path,
            "state_applied": self.mode in {"offline_test", "assisted_offline"},
            "osc_midi_sent": False,
            "tracks": [track.to_dict() for track in loaded.tracks],
            "channel_map": loaded.channel_map,
        }

    @staticmethod
    def _module_status(
        evaluations: dict[str, dict[str, dict[str, Any]]],
        *,
        source_separation: dict[str, Any] | None = None,
        action_planner: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        status: dict[str, Any] = {}
        for by_critic in evaluations.values():
            for name, result in by_critic.items():
                item = status.setdefault(
                    name,
                    {
                        "available": False,
                        "participated": True,
                        "role": result.get("role", ""),
                        "warnings": [],
                    },
                )
                item["available"] = bool(item["available"] or result.get("model_available", False))
                for warning in result.get("warnings", []):
                    if warning and warning not in item["warnings"]:
                        item["warnings"].append(warning)
        if source_separation is not None:
            status["demucs_or_openunmix"] = {
                "available": bool(source_separation.get("available", False)),
                "participated": bool(source_separation.get("participated", True)),
                "role": source_separation.get("role", "offline_source_separator"),
                "warnings": list(source_separation.get("warnings", [])),
                "backend": source_separation.get("backend", ""),
            }
        if action_planner is not None:
            status["automix_toolkit_fxnorm_diffmst_deepafx"] = {
                "available": bool(action_planner.get("available", True)),
                "participated": True,
                "role": action_planner.get("role", "candidate_action_planner_research_base"),
                "warnings": list(action_planner.get("warnings", [])),
                "technologies": list(action_planner.get("technologies", [])),
                "detected_reference_code": dict(action_planner.get("detected_reference_code", {})),
            }
        return status
