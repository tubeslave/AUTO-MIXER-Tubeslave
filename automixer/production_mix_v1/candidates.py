"""Staged candidate generation for `production_mix_v1`."""

from __future__ import annotations

from typing import Any, Mapping

from .config import ProductionMixConfig
from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_FX_SEND,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    MODE_LIVE,
    ROLE_ACTION_ROLE_SPECIFIC,
    ROLE_ACTION_UNIVERSAL_SAFE,
    AudioStem,
    MixAction,
    MixCandidate,
)


AYAIC_TARGETS = {
    "lead_vocal": -22.0,
    "backing_vocal": -27.0,
    "kick": -25.0,
    "snare": -25.0,
    "toms": -27.0,
    "bass": -25.0,
    "electric_guitar": -25.0,
    "acoustic_guitar": -25.0,
    "guitar": -25.0,
    "keys": -30.0,
    "synth": -30.0,
    "overheads": -30.0,
    "drum_room": -32.0,
    "hihat": -31.0,
    "ride": -31.0,
    "cymbal": -31.0,
    "playback": -30.0,
    "playback_music": -30.0,
    "music_stem": -30.0,
    "fx_return": -35.0,
    "unknown": -30.0,
}


class StagedCandidateGenerator:
    """Generate bounded staged candidates from baseline analysis."""

    def __init__(self, config: ProductionMixConfig):
        self.config = config

    def generate(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        *,
        mode: str,
    ) -> list[MixCandidate]:
        candidates = [
            MixCandidate(
                name=self.config.baseline_candidate_name,
                stage="baseline",
                source_modules=["production_mix_v1.candidates"],
                metadata={
                    "candidate_type": "utility",
                    "rationale": "No-change baseline for A/B scoring and rollback.",
                },
            )
        ]
        candidates.extend(self._ayaic_balance_candidate(stems, baseline_analysis, mode))
        candidates.extend(self._spectral_candidates(stems, baseline_analysis, mode))
        candidates.extend(self._dynamics_candidates(stems, baseline_analysis, mode))
        candidates.extend(self._combined_candidate(candidates, mode))
        candidates.extend(self._musical_recipe_candidates(stems, baseline_analysis, mode))
        return candidates[: self.config.max_candidates(mode)]

    def _ayaic_balance_candidate(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        mode: str,
    ) -> list[MixCandidate]:
        actions: list[MixAction] = []
        max_step = self._max_gain_step(mode)
        stem_metrics = baseline_analysis.get("stems", {})
        for stem in stems:
            if not _allows_role_specific(stem):
                continue
            metrics = dict(stem_metrics.get(str(stem.channel_id), {}).get("metrics", {}))
            measured = float(metrics.get("lufs_integrated_approx", metrics.get("rms_db", -100.0)))
            if measured <= -90.0:
                continue
            target = AYAIC_TARGETS.get(stem.role, AYAIC_TARGETS["unknown"])
            delta = max(-max_step, min(max_step, target - measured))
            if delta > 0.0 and _low_signal_from_metrics(metrics, self.config):
                continue
            if abs(delta) < 0.35:
                continue
            actions.append(
                MixAction(
                    action_type=ACTION_GAIN,
                    target=stem.name,
                    channel_id=stem.channel_id,
                    parameters={
                        "gain_db": round(delta, 3),
                        "target_lufs": target,
                        "measured_lufs": round(measured, 3),
                        "current_fader_db": stem.current_fader_db,
                        **_metric_params(metrics),
                        **_role_params(stem),
                        "reason": "source/stem static gain balance",
                        "expected_metric_change": "stem_lufs_moves_toward_role_target",
                    },
                    rationale=(
                        f"Ayaic level-plane balance: {stem.name} is {target - measured:+.2f} LU "
                        "from its source/stem target."
                    ),
                    stage="ayaic_balance",
                    source_modules=["ayaic_level_planes", "production_mix_v1"],
                )
            )
        if not actions:
            return []
        return [
            MixCandidate(
                name="candidate_001_ayaic_source_stem_balance",
                stage="ayaic_balance",
                actions=actions,
                source_modules=["production_mix_v1.candidates", "ayaic_level_planes"],
                metadata={
                    "candidate_type": "large_remix",
                    "rationale": "Bounded source/stem level-plane candidate.",
                },
            )
        ]

    def _spectral_candidates(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        mode: str,
    ) -> list[MixCandidate]:
        mix = baseline_analysis.get("mix", {})
        ratios = dict(mix.get("band_ratios", {}) or {})
        candidates: list[MixCandidate] = []
        if ratios.get("mud", 0.0) + ratios.get("boxiness", 0.0) > 0.28:
            culprit = self._culprit_for_band(stems, baseline_analysis, "mud")
            if culprit is not None:
                candidates.append(
                    self._eq_candidate(
                        "candidate_002_spectral_ceiling_mud",
                        culprit,
                        frequency_hz=260.0,
                        gain_db=max(self._max_eq_cut(mode), -1.4),
                        q=1.1,
                        reason="Mud/boxiness ratio is above the no-reference corridor.",
                    )
                )
        if ratios.get("harshness", 0.0) > 0.18:
            culprit = self._culprit_for_band(stems, baseline_analysis, "harshness")
            if culprit is not None:
                candidates.append(
                    self._eq_candidate(
                        "candidate_003_spectral_ceiling_harshness",
                        culprit,
                        frequency_hz=4200.0,
                        gain_db=max(self._max_eq_cut(mode), -1.2),
                        q=1.6,
                        reason="Harshness energy is high; use a bounded spectral guard cut.",
                    )
                )
        if ratios.get("sub", 0.0) > 0.20:
            culprit = self._culprit_for_band(stems, baseline_analysis, "sub")
            if culprit is not None:
                candidates.append(
                    self._eq_candidate(
                        "candidate_004_low_end_control",
                        culprit,
                        frequency_hz=60.0,
                        gain_db=max(self._max_eq_cut(mode), -1.0),
                        q=1.0,
                        reason="Sub energy is crowding headroom; candidate tightens low end.",
                    )
                )
        return candidates

    def _dynamics_candidates(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        mode: str,
    ) -> list[MixCandidate]:
        del mode
        stem_payload = baseline_analysis.get("stems", {})
        best: tuple[float, AudioStem] | None = None
        for stem in stems:
            if not _allows_role_specific(stem):
                continue
            metrics = stem_payload.get(str(stem.channel_id), {}).get("metrics", {})
            if _low_signal_from_metrics(metrics, self.config):
                continue
            crest = float(metrics.get("crest_factor_db", 0.0))
            if crest > 18.0 and (best is None or crest > best[0]):
                best = (crest, stem)
        if best is None:
            return []
        crest, stem = best
        action = MixAction(
            action_type=ACTION_COMPRESSION,
            target=stem.name,
            channel_id=stem.channel_id,
            parameters={
                "threshold_db": -18.0,
                "ratio": 1.6 if stem.role not in {"lead_vocal", "bass"} else 2.0,
                "attack_ms": 15.0,
                "release_ms": 160.0,
                "knee_db": 3.0,
                "makeup_gain_db": 0.0,
                "wet_mix_percent": 100.0,
                "target_gain_reduction_db": 0.8,
                **_metric_params(stem_payload.get(str(stem.channel_id), {}).get("metrics", {})),
                **_role_params(stem),
                "reason": f"{stem.name} crest factor is {crest:.2f} dB.",
                "expected_metric_change": "crest_factor_moves_toward_target_without_makeup_gain",
            },
            rationale=f"{stem.name} crest factor is {crest:.2f} dB; bounded compression candidate.",
            stage="dynamics_control",
            source_modules=["production_mix_v1.candidates"],
        )
        return [
            MixCandidate(
                name="candidate_005_bounded_crest_control",
                stage="dynamics_control",
                actions=[action],
                source_modules=["production_mix_v1.candidates"],
                metadata={
                    "candidate_type": "single_fix",
                    "rationale": "Gentle source compression before final gain.",
                },
            )
        ]

    @staticmethod
    def _combined_candidate(candidates: list[MixCandidate], mode: str) -> list[MixCandidate]:
        del mode
        balance = next((item for item in candidates if item.stage == "ayaic_balance"), None)
        spectral = [item for item in candidates if item.stage == "spectral_guard"][:2]
        if balance is None or not spectral:
            return []
        actions = [*balance.actions]
        for candidate in spectral:
            actions.extend(candidate.actions)
        return [
            MixCandidate(
                name="candidate_007_combined_balance_spectral_guard",
                stage="combined",
                actions=actions,
                source_modules=["production_mix_v1.candidates"],
                metadata={
                    "candidate_type": "large_remix",
                    "rationale": (
                        "Combined candidate applies Ayaic balance plus bounded spectral "
                        "guards for offline A/B scoring."
                    )
                },
            )
        ]

    def _musical_recipe_candidates(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        mode: str,
    ) -> list[MixCandidate]:
        if mode == MODE_LIVE and not bool(
            self.config.reference_recipe.get("enable_inherited_reference", True)
        ):
            return []
        inherited = str(
            self.config.reference_recipe.get(
                "inherited_reference_recipe",
                "FX15_SECTION_MOD_SPACE_AIR_MINUS_080",
            )
        )
        candidates: list[MixCandidate] = []

        fx15_actions = self._fx15_space_actions(stems)
        if fx15_actions:
            candidates.append(
                MixCandidate(
                    name="candidate_015_accepted_fx15_space_baseline",
                    stage="musical_recipe_fx15_space",
                    actions=fx15_actions,
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "reference_taste",
                        "inherited_reference_recipe": inherited,
                        "space_fx_actions": len(fx15_actions),
                        "rationale": "Accepted FX15 section/mod space recipe adapted as a musical baseline.",
                    },
                )
            )

        vocal_actions = [
            *self._fx_actions_for_roles(
                stems,
                {"lead_vocal", "backing_vocal"},
                "vocal_depth",
                stage="vocal_space_depth",
                reason="Vocal depth send adds front/back placement without raising fader level.",
            ),
            *self._compression_actions_for_roles(
                stems,
                {"lead_vocal", "backing_vocal"},
                "vocal_leveling",
                baseline_analysis=baseline_analysis,
                stage="vocal_leveling",
                reason="Soft vocal leveling improves density before any final gain.",
            ),
        ]
        if vocal_actions:
            candidates.append(
                MixCandidate(
                    name="candidate_025_vocal_space_and_depth",
                    stage="musical_recipe_vocal_depth",
                    actions=vocal_actions,
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "vocal_depth",
                        "space_fx_actions": len([a for a in vocal_actions if a.action_type == ACTION_FX_SEND]),
                        "dynamics_actions": len([a for a in vocal_actions if a.action_type == ACTION_COMPRESSION]),
                        "rationale": "Vocal space and gentle leveling candidate.",
                    },
                )
            )

        drum_roles = {"kick", "snare", "toms", "overheads", "drum_room", "hihat", "ride"}
        drum_actions = [
            *self._fx_actions_for_roles(
                stems,
                {"snare", "toms", "overheads", "drum_room", "hihat", "ride"},
                "drum_room",
                stage="drum_room",
                reason="Short drum room send restores kit depth and section glue.",
            ),
            *self._compression_actions_for_roles(
                stems,
                drum_roles,
                "drum_glue",
                baseline_analysis=baseline_analysis,
                stage="drum_glue",
                reason="Bounded drum glue compression controls crest without final limiting.",
            ),
        ]
        if drum_actions:
            candidates.append(
                MixCandidate(
                    name="candidate_035_drum_room_glue",
                    stage="musical_recipe_drum_room_glue",
                    actions=drum_actions,
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "drum_glue",
                        "space_fx_actions": len([a for a in drum_actions if a.action_type == ACTION_FX_SEND]),
                        "dynamics_actions": len([a for a in drum_actions if a.action_type == ACTION_COMPRESSION]),
                        "rationale": "Drum room and glue candidate.",
                    },
                )
            )

        parallel_actions = self._parallel_actions_for_roles(
            stems,
            drum_roles | {"bass"},
            baseline_analysis=baseline_analysis,
        )
        if parallel_actions:
            candidates.append(
                MixCandidate(
                    name="candidate_045_parallel_compression",
                    stage="musical_recipe_parallel_compression",
                    actions=parallel_actions,
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "drum_glue",
                        "dynamics_actions": len(parallel_actions),
                        "rationale": "Parallel compression candidate for subjective density without flattening dry path.",
                    },
                )
            )

        track_compression = [
            *self._compression_actions_for_roles(
                stems,
                {"lead_vocal", "backing_vocal"},
                "vocal_leveling",
                baseline_analysis=baseline_analysis,
                stage="track_compression_recipe",
                reason="Track compression recipe: vocal leveling.",
            ),
            *self._compression_actions_for_roles(
                stems,
                {"bass"},
                "bass_leveling",
                baseline_analysis=baseline_analysis,
                stage="track_compression_recipe",
                reason="Track compression recipe: bass leveling.",
            ),
            *self._compression_actions_for_roles(
                stems,
                drum_roles,
                "drum_glue",
                baseline_analysis=baseline_analysis,
                stage="track_compression_recipe",
                reason="Track compression recipe: drum glue.",
            ),
        ]
        if track_compression:
            candidates.append(
                MixCandidate(
                    name="candidate_055_track_compression_recipe",
                    stage="musical_recipe_track_compression",
                    actions=track_compression,
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "large_remix",
                        "dynamics_actions": len(track_compression),
                        "rationale": "Accepted track/stem compression recipe candidate.",
                    },
                )
            )

        air_action = self._final_air_action()
        if fx15_actions and air_action is not None:
            candidates.append(
                MixCandidate(
                    name="candidate_065_fx15_air_minus_080_reference",
                    stage="musical_recipe_fx15_air_reference",
                    actions=[*fx15_actions, air_action],
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "reference_taste",
                        "inherited_reference_recipe": inherited,
                        "space_fx_actions": len(fx15_actions),
                        "rationale": "Accepted reference recipe: FX15 section/mod space plus final air trim -0.80 dB.",
                    },
                )
            )

        static_gain_actions = self._static_gain_balance_actions(
            stems,
            baseline_analysis,
            mode,
            stage="static_gain_balance",
        )
        vocal_balance_actions = self._role_balance_actions(
            stems,
            baseline_analysis,
            {"lead_vocal", "backing_vocal"},
            target_lufs=-23.0,
            mode=mode,
            stage="vocal_balance",
            reason="Combined musical mix: vocal balance before space and dynamics.",
        )
        kick_bass_balance_actions = self._role_balance_actions(
            stems,
            baseline_analysis,
            {"kick", "bass"},
            target_lufs=-25.0,
            mode=mode,
            stage="kick_bass_balance",
            reason="Combined musical mix: kick/bass balance before low-end scoring.",
        )
        corrective_eq_actions = [
            action
            for candidate in self._spectral_candidates(stems, baseline_analysis, mode)
            for action in candidate.actions
        ][:2]
        combined_actions = [
            *static_gain_actions,
            *vocal_balance_actions,
            *kick_bass_balance_actions,
            *corrective_eq_actions,
            *fx15_actions,
            *self._compression_actions_for_roles(
                stems,
                {"lead_vocal", "backing_vocal"},
                "vocal_leveling",
                baseline_analysis=baseline_analysis,
                stage="combined_musical_recipe",
                reason="Combined musical mix: vocal leveling.",
            ),
            *self._compression_actions_for_roles(
                stems,
                {"bass"},
                "bass_leveling",
                baseline_analysis=baseline_analysis,
                stage="combined_musical_recipe",
                reason="Combined musical mix: bass leveling.",
            ),
            *self._compression_actions_for_roles(
                stems,
                drum_roles,
                "drum_glue",
                baseline_analysis=baseline_analysis,
                stage="combined_musical_recipe",
                reason="Combined musical mix: drum glue.",
            ),
            *self._parallel_actions_for_roles(
                stems,
                {"kick", "snare", "bass"},
                baseline_analysis=baseline_analysis,
            ),
        ]
        if air_action is not None:
            combined_actions.append(air_action)
        if combined_actions:
            candidates.append(
                MixCandidate(
                    name="candidate_075_combined_musical_mix",
                    stage="musical_recipe_combined",
                    actions=combined_actions,
                    source_modules=["production_mix_v1.musical_recipe_adapter"],
                    metadata={
                        "candidate_type": "large_remix",
                        "inherited_reference_recipe": inherited,
                        "static_gain_actions": len(static_gain_actions),
                        "fader_balance_actions": len(
                            static_gain_actions + vocal_balance_actions + kick_bass_balance_actions
                        ),
                        "space_fx_actions": len([a for a in combined_actions if a.action_type == ACTION_FX_SEND]),
                        "dynamics_actions": len(
                            [
                                a
                                for a in combined_actions
                                if a.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}
                            ]
                        ),
                        "rationale": "Combined musical mix candidate with space, compression, and final air reference.",
                    },
                )
            )
        return candidates

    def _static_gain_balance_actions(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        mode: str,
        *,
        stage: str,
    ) -> list[MixAction]:
        actions: list[MixAction] = []
        max_step = min(2.5, self._max_gain_step(mode))
        stem_metrics = baseline_analysis.get("stems", {})
        for stem in stems:
            if not _allows_role_specific(stem):
                continue
            metrics = dict(stem_metrics.get(str(stem.channel_id), {}).get("metrics", {}))
            measured = float(metrics.get("lufs_integrated_approx", metrics.get("rms_db", -100.0)))
            if measured <= -90.0:
                continue
            target = AYAIC_TARGETS.get(stem.role, AYAIC_TARGETS["unknown"])
            delta = max(-max_step, min(max_step, target - measured))
            if delta > 0.0 and _low_signal_from_metrics(metrics, self.config):
                continue
            if abs(delta) < 0.25:
                continue
            actions.append(
                MixAction(
                    action_type=ACTION_GAIN,
                    target=stem.name,
                    channel_id=stem.channel_id,
                    parameters={
                        "gain_db": round(delta, 3),
                        "target_lufs": target,
                        "measured_lufs": round(measured, 3),
                        "current_fader_db": stem.current_fader_db,
                        **_metric_params(metrics),
                        **_role_params(stem),
                        "reason": "static_gain_balance",
                        "expected_metric_change": "reduce_stem_loudness_spread",
                    },
                    rationale=(
                        f"Static gain balance: {stem.name} moves {delta:+.2f} dB "
                        f"toward role target {target:.1f} LUFS."
                    ),
                    stage=stage,
                    source_modules=["static_gain_balance_adapter", "production_mix_v1"],
                )
            )
        return actions

    def _role_balance_actions(
        self,
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        roles: set[str],
        *,
        target_lufs: float,
        mode: str,
        stage: str,
        reason: str,
    ) -> list[MixAction]:
        actions: list[MixAction] = []
        max_step = min(1.5, self._max_gain_step(mode))
        stem_metrics = baseline_analysis.get("stems", {})
        for stem in stems:
            if stem.role not in roles:
                continue
            if not _allows_role_specific(stem):
                continue
            metrics = dict(stem_metrics.get(str(stem.channel_id), {}).get("metrics", {}))
            measured = float(metrics.get("lufs_integrated_approx", metrics.get("rms_db", -100.0)))
            if measured <= -90.0:
                continue
            delta = max(-max_step, min(max_step, target_lufs - measured))
            if delta > 0.0 and _low_signal_from_metrics(metrics, self.config):
                continue
            if abs(delta) < 0.20:
                continue
            actions.append(
                MixAction(
                    action_type=ACTION_GAIN,
                    target=stem.name,
                    channel_id=stem.channel_id,
                    parameters={
                        "gain_db": round(delta, 3),
                        "target_lufs": target_lufs,
                        "measured_lufs": round(measured, 3),
                        "current_fader_db": stem.current_fader_db,
                        **_metric_params(metrics),
                        **_role_params(stem),
                        "reason": reason,
                        "expected_metric_change": "role_balance_moves_toward_mix_target",
                    },
                    rationale=f"{reason} Target: {stem.name}, delta {delta:+.2f} dB.",
                    stage=stage,
                    source_modules=["role_balance_adapter", "production_mix_v1"],
                )
            )
        return actions

    def _fx15_space_actions(self, stems: list[AudioStem]) -> list[MixAction]:
        return self._fx_actions_for_roles(
            stems,
            {
                "lead_vocal",
                "backing_vocal",
                "electric_guitar",
                "acoustic_guitar",
                "guitar",
                "keys",
                "synth",
                "overheads",
                "drum_room",
                "hihat",
                "ride",
                "playback",
                "music_stem",
            },
            "fx15_section_mod_space",
            stage="fx15_section_mod_space",
            reason="Inherited FX15 section/mod space send from accepted reference recipe.",
        )

    def _fx_actions_for_roles(
        self,
        stems: list[AudioStem],
        roles: set[str],
        recipe_key: str,
        *,
        stage: str,
        reason: str,
    ) -> list[MixAction]:
        recipe = dict(self.config.reference_recipe.get(recipe_key, {}) or {})
        actions: list[MixAction] = []
        for stem in stems:
            if stem.role not in roles:
                continue
            if not _allows_role_specific(stem):
                continue
            actions.append(
                MixAction(
                    action_type=ACTION_FX_SEND,
                    target=stem.name,
                    channel_id=stem.channel_id,
                    parameters={
                        "recipe": recipe_key,
                        "fx_slot": _fx_slot_for_recipe(recipe_key),
                        "fx_model": _fx_model_for_recipe(recipe_key),
                        "send_level_db": float(recipe.get("send_db", recipe.get("send_level_db", -18.0))),
                        "return_level_db": float(recipe.get("return_level_db", -12.0)),
                        "send_mode": str(recipe.get("send_mode", "post_fader")),
                        "route_to_main": bool(recipe.get("route_to_main", True)),
                        "wet_target_percent": float(recipe.get("wet_target_percent", 4.0)),
                        "pre_delay_ms": float(recipe.get("pre_delay_ms", 18.0)),
                        "decay_ms": float(recipe.get("decay_ms", 900.0)),
                        "width": float(recipe.get("width", 0.70)),
                        **_role_params(stem),
                        "reason": reason,
                    },
                    rationale=f"{reason} Target: {stem.name}.",
                    stage=stage,
                    source_modules=["accepted_reference_recipe_adapter", "production_mix_v1"],
                )
            )
        return actions

    def _compression_actions_for_roles(
        self,
        stems: list[AudioStem],
        roles: set[str],
        recipe_key: str,
        *,
        baseline_analysis: Mapping[str, Any],
        stage: str,
        reason: str,
    ) -> list[MixAction]:
        recipes = dict(self.config.reference_recipe.get("compression_recipe", {}) or {})
        recipe = dict(recipes.get(recipe_key, {}) or {})
        actions: list[MixAction] = []
        for stem in stems:
            if stem.role not in roles:
                continue
            if not _allows_role_specific(stem):
                continue
            metrics = dict(
                baseline_analysis.get("stems", {})
                .get(str(stem.channel_id), {})
                .get("metrics", {})
            )
            if _low_signal_from_metrics(metrics, self.config):
                continue
            stem_recipe = dict(recipe)
            if recipe_key == "drum_glue" and stem.role == "kick":
                stem_recipe.update(
                    {
                        "threshold_db": -18.0,
                        "ratio": 1.3,
                        "attack_ms": 20.0,
                        "release_ms": 140.0,
                        "wet_mix_percent": 55.0,
                        "target_gain_reduction_db": 0.25,
                    }
                )
            actions.append(
                MixAction(
                    action_type=ACTION_COMPRESSION,
                    target=stem.name,
                    channel_id=stem.channel_id,
                    parameters={
                        "threshold_db": float(stem_recipe.get("threshold_db", -21.0)),
                        "ratio": float(stem_recipe.get("ratio", 1.8)),
                        "attack_ms": float(stem_recipe.get("attack_ms", 18.0)),
                        "release_ms": float(stem_recipe.get("release_ms", 180.0)),
                        "knee_db": float(stem_recipe.get("knee_db", 3.0)),
                        "makeup_gain_db": float(
                            stem_recipe.get("makeup_gain_db", stem_recipe.get("makeup_db", 0.0))
                        ),
                        "wet_mix_percent": float(stem_recipe.get("wet_mix_percent", 100.0)),
                        "target_gain_reduction_db": float(
                            stem_recipe.get("target_gain_reduction_db", 0.8)
                        ),
                        **_metric_params(metrics),
                        **_role_params(stem),
                        "reason": reason,
                        "expected_metric_change": "measured_gain_reduction_proxy_above_threshold",
                    },
                    rationale=f"{reason} Target: {stem.name}.",
                    stage=stage,
                    source_modules=["track_compression_recipe_adapter", "production_mix_v1"],
                )
            )
        return actions

    def _parallel_actions_for_roles(
        self,
        stems: list[AudioStem],
        roles: set[str],
        *,
        baseline_analysis: Mapping[str, Any],
    ) -> list[MixAction]:
        recipes = dict(self.config.reference_recipe.get("compression_recipe", {}) or {})
        recipe = dict(recipes.get("parallel", {}) or {})
        actions: list[MixAction] = []
        for stem in stems:
            if stem.role not in roles:
                continue
            if not _allows_role_specific(stem):
                continue
            metrics = dict(
                baseline_analysis.get("stems", {})
                .get(str(stem.channel_id), {})
                .get("metrics", {})
            )
            if _low_signal_from_metrics(metrics, self.config):
                continue
            stem_recipe = dict(recipe)
            if stem.role == "kick":
                stem_recipe.update(
                    {
                        "threshold_db": -18.0,
                        "ratio": 2.0,
                        "wet_mix_percent": 8.0,
                        "target_gain_reduction_db": 0.15,
                    }
                )
            actions.append(
                MixAction(
                    action_type=ACTION_PARALLEL_COMPRESSION,
                    target=stem.name,
                    channel_id=stem.channel_id,
                    parameters={
                        "threshold_db": float(stem_recipe.get("threshold_db", -24.0)),
                        "ratio": float(stem_recipe.get("ratio", 3.2)),
                        "attack_ms": float(stem_recipe.get("attack_ms", 10.0)),
                        "release_ms": float(stem_recipe.get("release_ms", 220.0)),
                        "knee_db": float(stem_recipe.get("knee_db", 4.0)),
                        "makeup_gain_db": float(
                            stem_recipe.get("makeup_gain_db", stem_recipe.get("makeup_db", 0.0))
                        ),
                        "wet_mix_percent": float(
                            stem_recipe.get(
                                "wet_mix_percent",
                                float(stem_recipe.get("blend", 0.22)) * 100.0,
                            )
                        ),
                        "target_gain_reduction_db": float(
                            stem_recipe.get("target_gain_reduction_db", 1.0)
                        ),
                        **_metric_params(metrics),
                        **_role_params(stem),
                        "reason": "Parallel density path blended under dry source.",
                        "expected_metric_change": "measured_parallel_gain_reduction_proxy_above_threshold",
                    },
                    rationale=f"Parallel density path blended under dry source. Target: {stem.name}.",
                    stage="parallel_compression",
                    source_modules=["parallel_compression_recipe_adapter", "production_mix_v1"],
                )
            )
        return actions

    def _final_air_action(self) -> MixAction | None:
        gain_db = float(self.config.reference_recipe.get("final_air_trim_db", -0.80))
        if abs(gain_db) < 1e-6:
            return None
        return MixAction(
            action_type=ACTION_EQ,
            target="mix_bus_air",
            channel_id=None,
            parameters={
                "band": 4,
                "frequency_hz": 10000.0,
                "gain_db": gain_db,
                "q": 0.70,
                "curve": "final_air_trim",
            },
            rationale="Accepted reference final air trim AIR_MINUS_0_80 explicitly requested for this pipeline.",
            stage="final_air_minus_080",
            source_modules=["accepted_reference_recipe_adapter", "production_mix_v1"],
        )

    def _eq_candidate(
        self,
        name: str,
        stem: AudioStem,
        *,
        frequency_hz: float,
        gain_db: float,
        q: float,
        reason: str,
    ) -> MixCandidate:
        action = MixAction(
            action_type=ACTION_EQ,
            target=stem.name,
            channel_id=stem.channel_id,
            parameters={
                "band": _eq_band_for_frequency(frequency_hz),
                "frequency_hz": frequency_hz,
                "gain_db": gain_db,
                "q": q,
                **_role_params(stem),
            },
            rationale=f"{reason} Culprit candidate: {stem.name}.",
            stage="spectral_guard",
            source_modules=["spectral_ceiling_eq_adapter", "production_mix_v1"],
        )
        return MixCandidate(
            name=name,
            stage="spectral_guard",
            actions=[action],
            source_modules=["production_mix_v1.candidates", "spectral_ceiling_eq_adapter"],
            metadata={"candidate_type": "single_fix", "rationale": reason},
        )

    @staticmethod
    def _culprit_for_band(
        stems: list[AudioStem],
        baseline_analysis: Mapping[str, Any],
        band_name: str,
    ) -> AudioStem | None:
        stem_payload = baseline_analysis.get("stems", {})
        best: tuple[float, AudioStem] | None = None
        for stem in stems:
            if not _allows_universal_safe(stem):
                continue
            metrics = stem_payload.get(str(stem.channel_id), {}).get("metrics", {})
            value = float(metrics.get("band_energy_db", {}).get(band_name, -120.0))
            if best is None or value > best[0]:
                best = (value, stem)
        return best[1] if best is not None else None

    def _max_gain_step(self, mode: str) -> float:
        if mode == MODE_LIVE:
            return min(
                float(self.config.safety_limits.get("max_channel_fader_change_db_live", 1.5)),
                float(self.config.mode_config(mode).get("max_db_step", 0.7)),
            )
        return float(self.config.safety_limits.get("max_channel_fader_change_db_offline", 4.0))

    def _max_eq_cut(self, mode: str) -> float:
        if mode == MODE_LIVE:
            return float(self.config.safety_limits.get("max_eq_cut_db_live", -2.5))
        return float(self.config.safety_limits.get("max_eq_cut_db_offline", -4.0))


def _eq_band_for_frequency(freq_hz: float) -> int:
    if freq_hz < 180.0:
        return 1
    if freq_hz < 1200.0:
        return 2
    if freq_hz < 5500.0:
        return 3
    return 4


def _allows_role_specific(stem: AudioStem) -> bool:
    detection = stem.role_detection
    if detection is None:
        return stem.role not in {"unknown", "playback", "playback_music", "music_stem"}
    return detection.allowed_action_level == ROLE_ACTION_ROLE_SPECIFIC


def _allows_universal_safe(stem: AudioStem) -> bool:
    detection = stem.role_detection
    if detection is None:
        return True
    return detection.allowed_action_level in {ROLE_ACTION_ROLE_SPECIFIC, ROLE_ACTION_UNIVERSAL_SAFE}


def _role_params(stem: AudioStem) -> dict[str, object]:
    detection = stem.role_detection
    if detection is None:
        return {
            "role": stem.role,
            "role_confidence": 1.0 if stem.role != "unknown" else 0.0,
            "role_allowed_action_level": ROLE_ACTION_ROLE_SPECIFIC if stem.role != "unknown" else "no_op",
            "role_is_ambiguous": stem.role == "unknown",
        }
    return {
        "role": detection.role,
        "role_confidence": float(detection.confidence),
        "role_allowed_action_level": detection.allowed_action_level,
        "role_is_ambiguous": bool(detection.is_ambiguous),
    }


def _metric_params(metrics: Mapping[str, Any]) -> dict[str, object]:
    return {
        "metric_source": "baseline_analysis",
        "rms_db": round(float(metrics.get("rms_db", -120.0)), 3),
        "lufs_integrated_approx": round(
            float(metrics.get("lufs_integrated_approx", metrics.get("rms_db", -120.0))),
            3,
        ),
        "crest_factor_db": round(float(metrics.get("crest_factor_db", 0.0)), 3),
    }


def _low_signal_from_metrics(metrics: Mapping[str, Any], config: ProductionMixConfig) -> bool:
    cfg = dict(config.compression_safety or {})
    rms_limit = float(cfg.get("low_signal_rms_dbfs", -45.0))
    lufs_limit = float(cfg.get("low_signal_lufs", -45.0))
    rms = float(metrics.get("rms_db", -120.0))
    lufs = float(metrics.get("lufs_integrated_approx", rms))
    return rms < rms_limit or lufs < lufs_limit


def _fx_slot_for_recipe(recipe_key: str) -> str:
    return {
        "fx15_section_mod_space": "FX15",
        "vocal_depth": "FX16",
        "drum_room": "FX14",
    }.get(recipe_key, "FX15")


def _fx_model_for_recipe(recipe_key: str) -> str:
    return {
        "fx15_section_mod_space": "SECTION_MOD_SPACE",
        "vocal_depth": "VOCAL_DEPTH_PLATE_ROOM",
        "drum_room": "SHORT_DRUM_ROOM",
    }.get(recipe_key, "SECTION_MOD_SPACE")
