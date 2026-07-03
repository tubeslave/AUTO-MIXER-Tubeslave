"""Staged candidate generator for production_mix_v1."""

from __future__ import annotations

from typing import Any, Mapping

from .analyzers import guitar_forwardness_from_analysis
from .config import ProductionMixConfig
from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    ROLE_ACTION_ROLE_SPECIFIC,
    ROLE_ACTION_UNIVERSAL_SAFE,
    AudioStem,
    MixAction,
    MixCandidate,
)


DRUM_GLUE_ROLES = {"kick", "snare", "toms"}
OVERHEAD_ROLES = {"overheads", "cymbal", "ride", "hihat"}


class StagedCandidateGenerator:
    def __init__(self, config: ProductionMixConfig):
        self.config = config

    def generate(self, stems: list[AudioStem], baseline_analysis: Mapping[str, Any], *, mode: str) -> list[MixCandidate]:
        headroom_actions = self._pre_sum_headroom_actions(stems, baseline_analysis)
        candidates = [
            MixCandidate(
                name=self.config.baseline_candidate_name,
                stage="baseline",
                metadata={"candidate_type": "utility", "rationale": "No-change baseline."},
            )
        ]
        if headroom_actions:
            candidates.append(
                MixCandidate(
                    "candidate_002_pre_sum_headroom_gain_stage",
                    "static_gain_balance",
                    headroom_actions,
                    metadata={
                        "candidate_type": "critical_safety_fix",
                        "critical_safety_fix": True,
                        "rationale": "Uniform pre-sum gain staging prevents internal overshoot before musical processing.",
                    },
                )
            )
        candidates.extend(self._simple_spectral(stems, baseline_analysis, headroom_actions))
        candidates.extend(self._single_crest(stems, baseline_analysis, headroom_actions))
        candidates.extend(self._reference_candidates(stems, baseline_analysis, headroom_actions))
        candidates.extend(self._musical_dynamics(stems, headroom_actions))
        return candidates[: self.config.max_candidates(mode)]

    def _pre_sum_headroom_actions(self, stems: list[AudioStem], baseline_analysis: Mapping[str, Any]) -> list[MixAction]:
        true_peak = float(baseline_analysis.get("mix", {}).get("true_peak_dbfs", -120.0))
        target = float(self.config.safety_limits.get("pre_sum_target_true_peak_dbfs", -2.0))
        if true_peak <= target:
            return []
        gain_db = round(target - true_peak, 3)
        return [
            MixAction(
                ACTION_GAIN,
                stem.name,
                stem.channel_id,
                {
                    "gain_db": gain_db,
                    "gain_stage": "pre_sum_headroom",
                    "utility_gain_staging": True,
                    "baseline_true_peak_dbfs": round(true_peak, 3),
                    "target_pre_sum_true_peak_dbfs": target,
                    **_role_params(stem),
                    "reason": "Uniform pre-sum headroom stage; preserves balances while preventing internal bus overshoot.",
                },
                f"Uniform pre-sum headroom trim {gain_db:.2f} dB to prevent internal bus overshoot.",
                "static_gain_balance",
            )
            for stem in stems
        ]

    def _simple_spectral(self, stems: list[AudioStem], baseline_analysis: Mapping[str, Any], headroom_actions: list[MixAction]) -> list[MixCandidate]:
        mix = dict(baseline_analysis.get("mix", {}))
        ratios = dict(mix.get("band_ratios", {}) or {})
        if ratios.get("sub", 0.0) <= 0.20:
            return []
        stem = next((s for s in stems if s.role in {"kick", "bass"} and _allows_universal_safe(s)), None)
        if stem is None:
            return []
        action = MixAction(
            ACTION_EQ,
            stem.name,
            stem.channel_id,
            {"frequency_hz": 60.0, "gain_db": -1.0, **_role_params(stem), "reason": "low_end_control"},
            "Low-end control single fix.",
            "spectral_guard",
        )
        return [MixCandidate("candidate_004_low_end_control", "spectral_guard", [*headroom_actions, action], metadata={"candidate_type": "single_fix", "critical_safety_fix": bool(headroom_actions)})]

    def _single_crest(self, stems: list[AudioStem], baseline_analysis: Mapping[str, Any], headroom_actions: list[MixAction]) -> list[MixCandidate]:
        best: tuple[float, AudioStem] | None = None
        for stem in stems:
            if not _allows_role_specific(stem):
                continue
            metrics = dict(baseline_analysis.get("stems", {}).get(str(stem.channel_id), {}).get("metrics", {}) or {})
            if _low_signal(metrics, self.config) or metrics.get("analyzer_issues"):
                continue
            crest = float(metrics.get("crest_factor_db", 0.0))
            if crest > 18.0 and (best is None or crest > best[0]):
                best = (crest, stem)
        if best is None:
            return []
        crest, stem = best
        action = MixAction(
            ACTION_COMPRESSION,
            stem.name,
            stem.channel_id,
            {**_compression_params("drum_glue", self.config), **_metric_params(baseline_analysis, stem), **_role_params(stem), "reason": f"{stem.name} crest {crest:.2f} dB"},
            f"{stem.name} crest factor is {crest:.2f} dB.",
            "dynamics_control",
        )
        return [MixCandidate("candidate_005_bounded_crest_control", "dynamics_control", [*headroom_actions, action], metadata={"candidate_type": "single_fix", "critical_safety_fix": bool(headroom_actions)})]

    def _reference_candidates(self, stems: list[AudioStem], baseline_analysis: Mapping[str, Any], headroom_actions: list[MixAction]) -> list[MixCandidate]:
        fx_actions = [
            self._fx_action(stem)
            for stem in stems
            if stem.role in {"lead_vocal", "backing_vocal", "electric_guitar", "keys", "snare", "toms"} and _allows_fx_send(stem)
        ]
        fx_actions = [action for action in fx_actions if action is not None]
        if not fx_actions:
            return []
        air = MixAction(ACTION_EQ, "mix_bus_air", None, {"frequency_hz": 10000.0, "gain_db": -0.80, "q": 0.7, "reason": "final_air_minus_080"}, "Accepted reference final air trim.", "final_air_minus_080")
        candidates = [
            MixCandidate("candidate_015_accepted_fx15_space_baseline", "musical_recipe_fx15_space", [*headroom_actions, *fx_actions], metadata={"candidate_type": "reference_taste", "critical_safety_fix": bool(headroom_actions)}),
            MixCandidate("candidate_065_fx15_air_minus_080_reference", "musical_recipe_fx15_air_reference", [*headroom_actions, *fx_actions, air], metadata={"candidate_type": "reference_taste", "critical_safety_fix": bool(headroom_actions)}),
        ]
        guitar_actions = self._guitar_control_actions(stems, baseline_analysis)
        if guitar_actions:
            candidates.append(
                MixCandidate(
                    "candidate_066_fx15_reference_with_guitar_control",
                    "musical_recipe_fx15_air_reference_guitar_control",
                    [*headroom_actions, *fx_actions, air, *guitar_actions],
                    metadata={
                        "candidate_type": "reference_taste_guitar_control",
                        "inherits_reference_candidate": self.config.accepted_reference_candidate_name,
                        "critical_safety_fix": bool(headroom_actions),
                        "reference_preserved": True,
                    },
                )
            )
        return candidates

    def _musical_dynamics(self, stems: list[AudioStem], headroom_actions: list[MixAction]) -> list[MixCandidate]:
        drum_actions = [
            self._parallel_action(stem, "parallel_compression")
            for stem in stems
            if stem.role in DRUM_GLUE_ROLES and _allows_role_specific(stem)
        ]
        drum_actions = [a for a in drum_actions if a is not None]
        result: list[MixCandidate] = []
        if drum_actions:
            result.append(MixCandidate("candidate_045_parallel_compression", "musical_recipe_parallel_compression", [*headroom_actions, *drum_actions], metadata={"candidate_type": "drum_glue", "critical_safety_fix": bool(headroom_actions)}))
        track_actions = [
            self._compression_action(stem)
            for stem in stems
            if stem.role in {"lead_vocal", "bass", *DRUM_GLUE_ROLES} and _allows_role_specific(stem)
        ]
        track_actions = [a for a in track_actions if a is not None]
        if track_actions:
            result.append(MixCandidate("candidate_055_track_compression_recipe", "musical_recipe_track_compression", [*headroom_actions, *track_actions], metadata={"candidate_type": "large_remix", "critical_safety_fix": bool(headroom_actions)}))
        return result

    def _fx_action(self, stem: AudioStem) -> MixAction:
        recipe = dict(self.config.reference_recipe.get("fx15_section_mod_space", {}) or {})
        return MixAction(
            ACTION_FX_SEND,
            stem.name,
            stem.channel_id,
            {
                "fx_slot": "FX15",
                "fx_model": "SECTION_MOD_SPACE",
                "send_level_db": float(recipe.get("send_db", -18.0)),
                "return_level_db": float(recipe.get("return_level_db", -6.0)),
                "send_mode": str(recipe.get("send_mode", "post_fader")),
                "route_to_main": bool(recipe.get("route_to_main", True)),
                "wet_target_percent": float(recipe.get("wet_target_percent", 6.0)),
                "pre_delay_ms": float(recipe.get("pre_delay_ms", 18.0)),
                "decay_ms": float(recipe.get("decay_ms", 920.0)),
                "width": float(recipe.get("width", 0.72)),
                "modulation_depth_ms": float(recipe.get("modulation_depth_ms", 4.5)),
                "modulation_rate_hz": float(recipe.get("modulation_rate_hz", 0.32)),
                **_role_params(stem),
                "reason": "Inherited FX15 section/mod space.",
            },
            f"Inherited FX15 section/mod space. Target: {stem.name}.",
            "fx15_section_mod_space",
        )

    def _compression_action(self, stem: AudioStem) -> MixAction:
        return MixAction(ACTION_COMPRESSION, stem.name, stem.channel_id, {**_compression_params("drum_glue", self.config), **_role_params(stem), "reason": "Track compression recipe"}, f"Track compression recipe. Target: {stem.name}.", "track_compression_recipe")

    def _parallel_action(self, stem: AudioStem, stage: str) -> MixAction:
        return MixAction(ACTION_PARALLEL_COMPRESSION, stem.name, stem.channel_id, {**_compression_params("parallel", self.config), **_role_params(stem), "reason": "Parallel density path blended under dry source."}, f"Parallel density path blended under dry source. Target: {stem.name}.", stage)

    def _guitar_control_actions(self, stems: list[AudioStem], baseline_analysis: Mapping[str, Any]) -> list[MixAction]:
        cfg = dict(self.config.reference_recipe.get("guitar_control", {}) or {})
        if not bool(cfg.get("enabled", True)):
            return []
        forwardness = dict(baseline_analysis.get("guitar_forwardness") or guitar_forwardness_from_analysis(baseline_analysis, self.config))
        if not bool(forwardness.get("lead_vocal_present")) or not bool(forwardness.get("electric_guitars_present")):
            return []
        if not bool(forwardness.get("guitars_too_forward")):
            return []
        guitars = [stem for stem in stems if stem.role == "electric_guitar"]
        min_confidence = float(cfg.get("min_role_confidence", 0.85))
        if not guitars or any((stem.role_detection is None or stem.role_detection.confidence < min_confidence or not _allows_role_specific(stem)) for stem in guitars):
            return []
        default_trim = float(cfg.get("default_trim_db", -0.7))
        cap = -abs(float(cfg.get("max_reference_trim_db", -1.5)))
        trim_db = round(max(min(default_trim, -0.01), cap), 3)
        return [
            MixAction(
                ACTION_GAIN,
                stem.name,
                stem.channel_id,
                {
                    "gain_db": trim_db,
                    "guitar_control": True,
                    "guitar_pair_trim_db": trim_db,
                    "max_reference_trim_db": cap,
                    "lead_vocal_present": True,
                    "guitar_vocal_masking_index_before": forwardness.get("guitar_vocal_masking_index", 0.0),
                    "guitar_to_vocal_midrange_ratio_db": forwardness.get("guitar_to_vocal_midrange_ratio_db", 0.0),
                    **_role_params(stem),
                    "reason": "Small reference-mode guitar trim to reduce vocal masking while preserving FX15/Air taste.",
                },
                f"Small guitar trim {trim_db:.2f} dB to reduce vocal masking while preserving FX15/Air taste.",
                "guitar_forwardness_control",
            )
            for stem in guitars
        ]


def _compression_params(key: str, config: ProductionMixConfig) -> dict[str, float | str]:
    recipe = dict(config.reference_recipe.get("compression_recipe", {}).get(key, {}) or {})
    return {
        "threshold_db": float(recipe.get("threshold_db", -22.0)),
        "ratio": float(recipe.get("ratio", 2.0)),
        "attack_ms": float(recipe.get("attack_ms", 12.0)),
        "release_ms": float(recipe.get("release_ms", 180.0)),
        "knee_db": float(recipe.get("knee_db", 3.0)),
        "makeup_gain_db": float(recipe.get("makeup_gain_db", 0.0)),
        "wet_mix_percent": float(recipe.get("wet_mix_percent", 100.0)),
        "target_gain_reduction_db": float(recipe.get("target_gain_reduction_db", 0.8)),
        "expected_metric_change": "controlled_density_without_headroom_violation",
    }


def _allows_role_specific(stem: AudioStem) -> bool:
    return stem.role_detection is not None and stem.role_detection.allowed_action_level == ROLE_ACTION_ROLE_SPECIFIC


def _allows_universal_safe(stem: AudioStem) -> bool:
    return stem.role_detection is None or stem.role_detection.allowed_action_level in {ROLE_ACTION_ROLE_SPECIFIC, ROLE_ACTION_UNIVERSAL_SAFE}


def _allows_fx_send(stem: AudioStem) -> bool:
    return _allows_universal_safe(stem)


def _role_params(stem: AudioStem) -> dict[str, object]:
    detection = stem.role_detection
    return {
        "role": stem.role,
        "role_confidence": detection.confidence if detection else 0.0,
        "role_allowed_action_level": detection.allowed_action_level if detection else "no_op",
        "role_is_ambiguous": detection.is_ambiguous if detection else True,
    }


def _metric_params(analysis: Mapping[str, Any], stem: AudioStem) -> dict[str, object]:
    metrics = dict(analysis.get("stems", {}).get(str(stem.channel_id), {}).get("metrics", {}) or {})
    return {
        "metric_source": "baseline_analysis",
        "rms_db": float(metrics.get("rms_db", -120.0)),
        "lufs_integrated_approx": float(metrics.get("lufs_integrated_approx", -120.0)),
        "crest_factor_db": float(metrics.get("crest_factor_db", 0.0)),
    }


def _low_signal(metrics: Mapping[str, Any], config: ProductionMixConfig) -> bool:
    return (
        float(metrics.get("rms_db", -120.0)) < float(config.compression_safety.get("low_signal_rms_dbfs", -45.0))
        or float(metrics.get("lufs_integrated_approx", -120.0)) < float(config.compression_safety.get("low_signal_lufs", -45.0))
    )
