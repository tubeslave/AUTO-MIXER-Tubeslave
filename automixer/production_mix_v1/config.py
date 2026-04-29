"""Config loader for `production_mix_v1`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "pipelines" / "production_mix_v1.yaml"
DEFAULT_ROLE_OVERRIDES_PATH = REPO_ROOT / "config" / "role_overrides.yaml"


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "production_mix_v1",
    "version": 1,
    "mode_defaults": {
        "offline": {
            "min_score_improvement": 0.002,
            "min_muq_delta": 0.0005,
            "min_material_final_delta_when_muq_unavailable": 0.010,
            "min_material_musical_delta_when_muq_unavailable": 0.020,
            "allow_small_improvements": True,
            "max_candidates": 24,
        },
        "live": {
            "min_score_improvement": 0.010,
            "min_muq_delta": 0.002,
            "allow_small_improvements": False,
            "max_actions_per_minute": 6,
            "max_db_step": 0.7,
            "max_eq_gain_step": 1.0,
            "cooldown_seconds_per_channel": 20,
            "max_candidates": 18,
        },
    },
    "reference_recipe": {
        "inherited_reference_recipe": "FX15_SECTION_MOD_SPACE_AIR_MINUS_080",
        "enable_inherited_reference": True,
        "transfer_scope": {
            "allow_fx_taste": True,
            "allow_air_trim": True,
            "allow_limiter_target": True,
            "allow_gentle_bus_glue": True,
            "allow_fixed_group_targets": False,
            "allow_channel_specific_compression_gr": False,
            "allow_unverified_pan_map": False,
        },
        "fx15_section_mod_space": {
            "send_db": -18.0,
            "return_level_db": -6.0,
            "send_mode": "post_fader",
            "route_to_main": True,
            "wet_target_percent": 6.0,
            "pre_delay_ms": 18.0,
            "decay_ms": 920.0,
            "width": 0.72,
        },
        "vocal_depth": {
            "send_db": -16.5,
            "return_level_db": -7.0,
            "send_mode": "post_fader",
            "route_to_main": True,
            "wet_target_percent": 7.0,
            "pre_delay_ms": 24.0,
            "decay_ms": 1180.0,
            "width": 0.62,
        },
        "drum_room": {
            "send_db": -19.5,
            "return_level_db": -8.0,
            "send_mode": "post_fader",
            "route_to_main": True,
            "wet_target_percent": 5.0,
            "pre_delay_ms": 8.0,
            "decay_ms": 760.0,
            "width": 0.80,
        },
        "final_air_trim_db": -0.80,
        "compression_recipe": {
            "vocal_leveling": {
                "threshold_db": -21.0,
                "ratio": 1.8,
                "attack_ms": 18.0,
                "release_ms": 180.0,
                "knee_db": 3.0,
                "makeup_gain_db": 0.0,
                "wet_mix_percent": 100.0,
                "target_gain_reduction_db": 0.8,
            },
            "drum_glue": {
                "threshold_db": -22.0,
                "ratio": 1.6,
                "attack_ms": 14.0,
                "release_ms": 180.0,
                "knee_db": 3.0,
                "makeup_gain_db": 0.0,
                "wet_mix_percent": 100.0,
                "target_gain_reduction_db": 0.8,
            },
            "bass_leveling": {
                "threshold_db": -20.0,
                "ratio": 2.0,
                "attack_ms": 22.0,
                "release_ms": 220.0,
                "knee_db": 3.0,
                "makeup_gain_db": 0.0,
                "wet_mix_percent": 100.0,
                "target_gain_reduction_db": 0.9,
            },
            "parallel": {
                "threshold_db": -24.0,
                "ratio": 3.2,
                "attack_ms": 10.0,
                "release_ms": 220.0,
                "knee_db": 4.0,
                "wet_mix_percent": 22.0,
                "makeup_gain_db": 0.0,
                "target_gain_reduction_db": 1.0,
            },
        },
        "scoring": {
            "dry_mix_penalty": 0.060,
            "overdynamic_penalty": 0.060,
            "fx_wetness_threshold": 0.002,
            "compression_gain_reduction_threshold_db": 0.15,
            "action_count_penalty_per_action_over_10": 0.004,
            "large_gain_penalty_per_db_over_2": 0.010,
            "dryness_penalty_weight": 0.080,
            "inherited_recipe_bonus": 0.045,
            "space_action_bonus": 0.018,
            "dynamics_action_bonus": 0.014,
            "final_air_reference_bonus": 0.012,
            "crest_density_bonus": 0.020,
            "max_recipe_bonus": 0.120,
        },
        "safety_adapter": {
            "max_space_vocal_clarity_drop": 0.100,
        },
    },
    "weights": {
        "musical_quality": 0.30,
        "vocal_clarity": 0.25,
        "spectral_balance": 0.15,
        "dynamics_control": 0.10,
        "low_end_control": 0.10,
        "translation_score": 0.05,
        "safety_margin": 0.05,
    },
    "critic_weights": {
        "muq_eval": 0.45,
        "mert": 0.20,
        "essentia": 0.20,
        "rules": 0.15,
        "audiobox_aesthetics_proxy": 0.00,
        "clap_proxy": 0.00,
        "panns_or_beats_proxy": 0.00,
    },
    "bands": {
        "sub": [40, 80],
        "punch": [80, 160],
        "mud": [160, 350],
        "boxiness": [350, 700],
        "intelligibility": [700, 1500],
        "presence": [1500, 3000],
        "harshness": [3000, 6000],
        "air": [6000, 10000],
        "polish": [10000, 16000],
    },
    "safety_limits": {
        "max_channel_fader_change_db_offline": 4.0,
        "max_channel_fader_change_db_live": 1.5,
        "max_eq_cut_db_offline": -4.0,
        "max_eq_boost_db_offline": 2.0,
        "max_eq_cut_db_live": -2.5,
        "max_eq_boost_db_live": 1.0,
        "min_phase_correlation": 0.2,
        "max_true_peak_dbfs": -1.0,
        "reject_if_vocal_clarity_drops": True,
        "reject_if_low_end_collapses": True,
        "reject_if_harshness_increases_too_much": True,
    },
    "decision": {
        "baseline_candidate_name": "candidate_000_no_change",
        "reject_proxy_only_winners": True,
        "require_safety_pass": True,
        "require_no_critical_degradation": True,
        "enable_rollback_snapshot": True,
    },
    "role_detection": {
        "role_specific_confidence": 0.85,
        "universal_safe_confidence": 0.60,
        "playback_role_specific_enabled": False,
    },
    "large_correction_guard": {
        "review_gain_db": 2.5,
        "reject_gain_db": 4.0,
        "max_gain_without_muq_db": 1.5,
        "max_actions_without_muq": 10,
        "group_gain_reject_db": 6.0,
        "many_large_moves_threshold_db": 2.0,
        "many_large_moves_count": 6,
        "ambience_or_playback_cut_review_db": -4.0,
        "drum_boost_review_db": 4.0,
    },
    "dryness_guard": {
        "ambience_roles": ["drum_room", "overheads", "hihat", "ride", "cymbal", "playback"],
        "ambience_cut_penalty_db": -4.0,
        "min_natural_ambience_preservation_score": 0.65,
    },
    "compression_safety": {
        "vocal_review_gr_db": 5.0,
        "rhythm_review_gr_db": 6.0,
        "any_reject_gr_db": 7.0,
        "low_signal_rms_dbfs": -45.0,
        "low_signal_lufs": -45.0,
        "metrics_conflict_threshold_db": 3.0,
    },
    "role_overrides": {"patterns": {}},
}


@dataclass(frozen=True)
class ProductionMixConfig:
    """Typed runtime config for production mix v1."""

    name: str = "production_mix_v1"
    version: int = 1
    mode_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    critic_weights: dict[str, float] = field(default_factory=dict)
    reference_recipe: dict[str, Any] = field(default_factory=dict)
    bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    safety_limits: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    role_detection: dict[str, Any] = field(default_factory=dict)
    large_correction_guard: dict[str, Any] = field(default_factory=dict)
    dryness_guard: dict[str, Any] = field(default_factory=dict)
    compression_safety: dict[str, Any] = field(default_factory=dict)
    role_overrides: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any] | None,
        *,
        warnings: list[str] | None = None,
    ) -> "ProductionMixConfig":
        merged = _deep_merge(DEFAULT_CONFIG, dict(payload or {}))
        return cls(
            name=str(merged.get("name", "production_mix_v1")),
            version=int(merged.get("version", 1)),
            mode_defaults={
                str(mode): dict(values or {})
                for mode, values in dict(merged.get("mode_defaults", {})).items()
            },
            weights=_float_mapping(merged.get("weights", {})),
            critic_weights=_float_mapping(merged.get("critic_weights", {})),
            reference_recipe=dict(merged.get("reference_recipe", {})),
            bands=_bands_mapping(merged.get("bands", {})),
            safety_limits=dict(merged.get("safety_limits", {})),
            decision=dict(merged.get("decision", {})),
            role_detection=dict(merged.get("role_detection", {})),
            large_correction_guard=dict(merged.get("large_correction_guard", {})),
            dryness_guard=dict(merged.get("dryness_guard", {})),
            compression_safety=dict(merged.get("compression_safety", {})),
            role_overrides=dict(merged.get("role_overrides", {})),
            warnings=tuple(warnings or ()),
        )

    def mode_config(self, mode: str) -> dict[str, Any]:
        return dict(self.mode_defaults.get(str(mode), self.mode_defaults.get("offline", {})))

    def max_candidates(self, mode: str) -> int:
        return int(self.mode_config(mode).get("max_candidates", 12))

    def min_score_improvement(self, mode: str) -> float:
        return float(self.mode_config(mode).get("min_score_improvement", 0.002))

    def min_muq_delta(self, mode: str) -> float:
        return float(self.mode_config(mode).get("min_muq_delta", 0.0005))

    @property
    def baseline_candidate_name(self) -> str:
        return str(self.decision.get("baseline_candidate_name", "candidate_000_no_change"))


def load_production_mix_config(path: str | Path | None = None) -> ProductionMixConfig:
    """Load the pipeline YAML config, falling back to built-in defaults."""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    warnings: list[str] = []
    if not config_path.exists():
        warnings.append(f"config_missing:{config_path}")
        return ProductionMixConfig.from_mapping({}, warnings=warnings)
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        warnings.append(f"config_load_failed:{exc}")
        payload = {}
    role_overrides = _load_role_overrides(DEFAULT_ROLE_OVERRIDES_PATH, warnings)
    if role_overrides:
        payload = _deep_merge(payload, {"role_overrides": role_overrides})
    return ProductionMixConfig.from_mapping(payload, warnings=warnings)


def _load_role_overrides(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        warnings.append(f"role_overrides_load_failed:{exc}")
    return {}


def _float_mapping(payload: Any) -> dict[str, float]:
    return {str(key): float(value) for key, value in dict(payload or {}).items()}


def _bands_mapping(payload: Any) -> dict[str, tuple[float, float]]:
    bands: dict[str, tuple[float, float]] = {}
    for key, value in dict(payload or {}).items():
        try:
            low, high = list(value)[:2]
            bands[str(key)] = (float(low), float(high))
        except (TypeError, ValueError):
            continue
    return bands


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
