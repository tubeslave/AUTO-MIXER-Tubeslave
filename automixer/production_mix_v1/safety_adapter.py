"""Safety governor for production_mix_v1."""

from __future__ import annotations

from typing import Any, Mapping

from .config import ProductionMixConfig
from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_PARALLEL_COMPRESSION,
    ROLE_ACTION_ROLE_SPECIFIC,
    MixAction,
    MixCandidate,
    SafetyResult,
)


OVERHEAD_ROLES = {"overheads", "cymbal", "ride", "hihat"}


class ProductionSafetyGovernor:
    def __init__(self, config: ProductionMixConfig):
        self.config = config

    def evaluate(self, candidate: MixCandidate, *, candidate_analysis: Mapping[str, Any], baseline_analysis: Mapping[str, Any], mode: str, dry_run: bool) -> SafetyResult:
        del mode, dry_run
        blocked: list[dict[str, Any]] = []
        rationale: list[str] = []
        true_peak = float(candidate_analysis.get("mix", candidate_analysis).get("true_peak_dbfs", -120.0))
        if true_peak > float(self.config.safety_limits.get("max_true_peak_dbfs", -1.0)):
            blocked.append({"reason": "true_peak_limit_exceeded", "true_peak_dbfs": true_peak})
        self._action_contracts(candidate, baseline_analysis, blocked)
        passed = not blocked
        if passed:
            rationale.append("Production Safety Governor passed.")
        return SafetyResult(candidate.name, passed=passed, allowed_actions=list(candidate.actions) if passed else [], blocked_actions=blocked, rationale=rationale)

    def _action_contracts(self, candidate: MixCandidate, baseline_analysis: Mapping[str, Any], blocked: list[dict[str, Any]]) -> None:
        for action in candidate.actions:
            role = _action_role(action)
            if action.channel_id is not None and _requires_role_specific(action) and action.parameters.get("role_allowed_action_level") != ROLE_ACTION_ROLE_SPECIFIC:
                blocked.append({"reason": "role_specific_action_blocked_by_confidence_gate", "action": action.to_dict()})
            if action.action_type == ACTION_PARALLEL_COMPRESSION and role in OVERHEAD_ROLES and not bool(self.config.compression_safety.get("allow_parallel_compression_on_overheads", False)):
                blocked.append({"reason": "review_required_parallel_compression_on_overheads", "role": role, "action": action.to_dict()})
            if candidate.metadata.get("candidate_type") == "drum_glue" and role == "bass":
                blocked.append({"reason": "drum_glue_contains_bass_requires_rhythm_section_glue", "action": action.to_dict()})
            if action.parameters.get("guitar_control"):
                cfg = dict(self.config.reference_recipe.get("guitar_control", {}) or {})
                max_trim = -abs(float(cfg.get("max_reference_trim_db", -1.5)))
                min_confidence = float(cfg.get("min_role_confidence", 0.85))
                gain_db = float(action.parameters.get("gain_db", 0.0))
                if role != "electric_guitar":
                    blocked.append({"reason": "guitar_control_target_not_electric_guitar", "action": action.to_dict()})
                if gain_db < max_trim or gain_db > 0.0:
                    blocked.append({"reason": "guitar_control_trim_out_of_bounds", "max_reference_trim_db": max_trim, "gain_db": gain_db, "action": action.to_dict()})
                if float(action.parameters.get("role_confidence", 0.0)) < min_confidence:
                    blocked.append({"reason": "guitar_control_blocked_by_low_role_confidence", "min_confidence": min_confidence, "action": action.to_dict()})
                if not bool(action.parameters.get("lead_vocal_present", False)):
                    blocked.append({"reason": "guitar_control_requires_lead_vocal", "action": action.to_dict()})
            if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION, ACTION_EQ, ACTION_GAIN} and action.channel_id is not None:
                metrics = dict(baseline_analysis.get("stems", {}).get(str(action.channel_id), {}).get("metrics", {}) or {})
                if metrics.get("analyzer_issues") and _requires_role_specific(action):
                    blocked.append({"reason": "analyzer_failure_blocks_role_specific_action", "issues": metrics.get("analyzer_issues"), "action": action.to_dict()})
            if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}:
                missing = [key for key in ("threshold_db", "ratio", "attack_ms", "release_ms", "knee_db", "makeup_gain_db", "wet_mix_percent", "target_gain_reduction_db", "reason", "expected_metric_change") if action.parameters.get(key) in (None, "")]
                if missing:
                    blocked.append({"reason": "compression_parameters_missing", "missing": missing, "action": action.to_dict()})


def _requires_role_specific(action: MixAction) -> bool:
    if action.action_type in {ACTION_COMPRESSION, ACTION_PARALLEL_COMPRESSION}:
        return True
    if action.action_type == ACTION_EQ:
        return float(action.parameters.get("gain_db", 0.0)) > 0.0
    if action.action_type == ACTION_GAIN and action.channel_id is not None:
        return float(action.parameters.get("gain_db", 0.0)) > 0.0
    return False


def _action_role(action: MixAction) -> str:
    if action.parameters.get("role"):
        return str(action.parameters["role"])
    text = str(action.target).lower()
    if "oh" in text or "overhead" in text:
        return "overheads"
    if "ride" in text:
        return "ride"
    if "hat" in text:
        return "hihat"
    if "bass" in text:
        return "bass"
    if "kick" in text:
        return "kick"
    if "snare" in text:
        return "snare"
    if "tom" in text:
        return "toms"
    return "unknown"
