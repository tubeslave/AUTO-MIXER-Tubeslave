"""Config loader for production_mix_v1."""

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
            "min_verified_reference_final_delta_when_muq_unavailable": 0.001,
            "min_verified_reference_musical_delta_when_muq_unavailable": 0.001,
            "win_margin_over_reference_when_muq_unavailable": 0.010,
            "reference_tiebreak_final_epsilon": 0.005,
            "reference_tiebreak_musical_epsilon": 0.005,
            "allow_small_improvements": True,
            "max_candidates": 24,
        },
        "live": {
            "min_score_improvement": 0.010,
            "min_muq_delta": 0.002,
            "allow_small_improvements": False,
            "max_actions_per_minute": 6,
            "max_candidates": 18,
        },
    },
    "reference_recipe": {
        "inherited_reference_recipe": "FX15_SECTION_MOD_SPACE_AIR_MINUS_080",
        "enable_inherited_reference": True,
        "final_air_trim_db": -0.80,
        "fx15_section_mod_space": {
            "send_db": -18.0,
            "return_level_db": -6.0,
            "send_mode": "post_fader",
            "route_to_main": True,
            "wet_target_percent": 6.0,
            "pre_delay_ms": 18.0,
            "decay_ms": 920.0,
            "width": 0.72,
            "modulation_depth_ms": 4.5,
            "modulation_rate_hz": 0.32,
        },
        "compression_recipe": {
            "vocal_leveling": {"threshold_db": -21.0, "ratio": 1.8, "attack_ms": 18.0, "release_ms": 180.0, "knee_db": 3.0, "makeup_gain_db": 0.0, "wet_mix_percent": 100.0, "target_gain_reduction_db": 0.8},
            "drum_glue": {"threshold_db": -22.0, "ratio": 1.6, "attack_ms": 14.0, "release_ms": 180.0, "knee_db": 3.0, "makeup_gain_db": 0.0, "wet_mix_percent": 100.0, "target_gain_reduction_db": 0.8},
            "parallel": {"threshold_db": -24.0, "ratio": 3.2, "attack_ms": 10.0, "release_ms": 220.0, "knee_db": 4.0, "makeup_gain_db": 0.0, "wet_mix_percent": 22.0, "target_gain_reduction_db": 1.0},
        },
        "scoring": {
            "fx_wetness_threshold": 0.002,
            "spatial_reflection_threshold": 0.001,
            "modulation_energy_threshold": 0.00015,
            "compression_gain_reduction_threshold_db": 0.15,
            "action_count_penalty": 0.0015,
            "overhead_dynamics_penalty": 0.020,
            "bass_in_drum_glue_penalty": 0.015,
            "reference_taste_bias": 0.003,
            "guitar_too_forward_penalty": 0.018,
            "guitar_vocal_masking_penalty": 0.014,
            "guitar_masking_improvement_reward": 0.026,
            "guitar_power_loss_penalty": 0.030,
        },
        "guitar_control": {
            "enabled": True,
            "default_trim_db": -0.7,
            "max_reference_trim_db": -1.5,
            "min_role_confidence": 0.85,
            "presence_ratio_threshold_db": 2.0,
            "midrange_ratio_threshold_db": -3.0,
            "masking_index_threshold": 0.62,
            "max_power_loss_db": 2.25,
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
        "rules": 1.0,
        "muq_eval": 0.0,
        "mert": 0.0,
        "essentia": 0.0,
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
        "min_phase_correlation": 0.2,
        "max_true_peak_dbfs": -1.0,
        "pre_sum_target_true_peak_dbfs": -2.0,
    },
    "autoeq": {
        "enabled": False,
        "mode": "offline",
        "safety_profile": "conservative",
        "apply_static_eq": True,
        "apply_dynamic_eq": True,
        "apply_group_eq": True,
        "apply_master_eq": False,
        "report_only": False,
        "osc_dry_run": True,
        "osc_apply": False,
        "live_limits": {
            "max_static_gain_db": 1.5,
            "max_dynamic_cut_db": 3.0,
            "max_dynamic_boost_db": 1.0,
            "max_master_eq_db": 0.75,
            "max_filters_per_channel": 3,
            "max_dynamic_filters_per_channel": 2,
            "min_confidence_to_apply": 0.6,
            "half_strength_confidence_until": 0.8,
            "smoothing_ms": 1000,
            "hold_time_ms": 2000,
        },
        "offline_limits": {
            "max_static_gain_db": 3.0,
            "max_dynamic_cut_db": 5.0,
            "max_dynamic_boost_db": 1.5,
            "max_master_eq_db": 1.5,
            "max_filters_per_channel": 5,
            "max_dynamic_filters_per_channel": 3,
            "min_confidence_to_apply": 0.55,
        },
        "priorities": {
            "lead_vocal": 100,
            "kick": 90,
            "snare": 85,
            "bass": 85,
            "back_vocal": 65,
            "guitar": 55,
            "accordion": 55,
            "playback": 50,
            "overheads": 45,
            "cymbals": 40,
        },
    },
    "decision": {
        "baseline_candidate_name": "candidate_000_no_change",
        "accepted_reference_candidate_name": "candidate_065_fx15_air_minus_080_reference",
        "reject_proxy_only_winners": True,
        "require_safety_pass": True,
        "require_reference_material_improvement": True,
    },
    "role_detection": {
        "role_specific_confidence": 0.85,
        "universal_safe_confidence": 0.60,
        "playback_role_specific_enabled": False,
    },
    "compression_safety": {
        "low_signal_rms_dbfs": -45.0,
        "low_signal_lufs": -45.0,
        "metrics_conflict_threshold_db": 3.0,
        "allow_parallel_compression_on_overheads": False,
        "internal_true_peak_review_dbfs": 6.0,
        "internal_true_peak_reject_dbfs": 12.0,
    },
    "dryness_guard": {
        "ambience_roles": ["drum_room", "overheads", "hihat", "ride", "cymbal", "music_stem"],
        "min_natural_ambience_preservation_score": 0.65,
        "ambience_cut_penalty_db": -4.0,
    },
    "role_overrides": {"patterns": {}},
}


@dataclass(frozen=True)
class ProductionMixConfig:
    name: str = "production_mix_v1"
    version: int = 1
    mode_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    critic_weights: dict[str, float] = field(default_factory=dict)
    reference_recipe: dict[str, Any] = field(default_factory=dict)
    bands: dict[str, tuple[float, float]] = field(default_factory=dict)
    safety_limits: dict[str, Any] = field(default_factory=dict)
    autoeq: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    role_detection: dict[str, Any] = field(default_factory=dict)
    compression_safety: dict[str, Any] = field(default_factory=dict)
    dryness_guard: dict[str, Any] = field(default_factory=dict)
    role_overrides: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None, *, warnings: list[str] | None = None) -> "ProductionMixConfig":
        merged = _deep_merge(DEFAULT_CONFIG, dict(payload or {}))
        return cls(
            name=str(merged.get("name", "production_mix_v1")),
            version=int(merged.get("version", 1)),
            mode_defaults={str(k): dict(v or {}) for k, v in dict(merged.get("mode_defaults", {})).items()},
            weights={str(k): float(v) for k, v in dict(merged.get("weights", {})).items()},
            critic_weights={str(k): float(v) for k, v in dict(merged.get("critic_weights", {})).items()},
            reference_recipe=dict(merged.get("reference_recipe", {})),
            bands={str(k): (float(v[0]), float(v[1])) for k, v in dict(merged.get("bands", {})).items()},
            safety_limits=dict(merged.get("safety_limits", {})),
            autoeq=dict(merged.get("autoeq", {})),
            decision=dict(merged.get("decision", {})),
            role_detection=dict(merged.get("role_detection", {})),
            compression_safety=dict(merged.get("compression_safety", {})),
            dryness_guard=dict(merged.get("dryness_guard", {})),
            role_overrides=dict(merged.get("role_overrides", {})),
            warnings=tuple(warnings or ()),
        )

    def mode_config(self, mode: str) -> dict[str, Any]:
        return dict(self.mode_defaults.get(mode, self.mode_defaults.get("offline", {})))

    def max_candidates(self, mode: str) -> int:
        return int(self.mode_config(mode).get("max_candidates", 12))

    def min_score_improvement(self, mode: str) -> float:
        return float(self.mode_config(mode).get("min_score_improvement", 0.002))

    def min_muq_delta(self, mode: str) -> float:
        return float(self.mode_config(mode).get("min_muq_delta", 0.0))

    @property
    def baseline_candidate_name(self) -> str:
        return str(self.decision.get("baseline_candidate_name", "candidate_000_no_change"))

    @property
    def accepted_reference_candidate_name(self) -> str:
        return str(self.decision.get("accepted_reference_candidate_name", "candidate_065_fx15_air_minus_080_reference"))


def load_production_mix_config(path: str | Path | None = None) -> ProductionMixConfig:
    warnings: list[str] = []
    payload: dict[str, Any] = {}
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        try:
            import yaml
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            warnings.append(f"config_load_failed:{exc}")
    else:
        warnings.append(f"config_missing:{config_path}")
    role_overrides = _load_yaml(DEFAULT_ROLE_OVERRIDES_PATH, warnings)
    if role_overrides:
        payload = _deep_merge(payload, {"role_overrides": role_overrides})
    return ProductionMixConfig.from_mapping(payload, warnings=warnings)


def _load_yaml(path: Path, warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception as exc:
        warnings.append(f"yaml_load_failed:{path}:{exc}")
        return {}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in base.items()}
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
