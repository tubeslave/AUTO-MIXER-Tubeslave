"""Explainable Decision Engine v2.

This module deliberately has no OSC or mixer-client imports. It turns analyzer
evidence, critic feedback and knowledge rules into an ``ActionPlan`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

from automixer.analyzer import normalize_analyzer_output
from automixer.critics import normalize_critic_evaluations
from automixer.knowledge import MixingKnowledgeBase, normalize_role

from .models import (
    ACTION_COMPRESSION,
    ACTION_EQ,
    ACTION_GAIN,
    ACTION_NO_ACTION,
    ACTION_PAN,
    RISK_CRITICAL,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    ActionDecision,
    ActionPlan,
    clamp_confidence,
)


@dataclass(frozen=True)
class DecisionEngineConfig:
    """Runtime knobs for conservative decision planning."""

    min_confidence: float = 0.55
    max_actions_per_plan: int = 8
    level_tolerance_lu: float = 1.5
    max_gain_recommendation_db: float = 3.0
    tonal_issue_threshold_db: float = 2.5
    max_eq_recommendation_db: float = 2.0
    crest_factor_threshold_db: float = 18.0
    dynamic_range_threshold_db: float = 18.0
    max_pan_recommendation: float = 0.2
    true_peak_boost_ceiling_dbtp: float = -3.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "DecisionEngineConfig":
        payload = payload or {}
        return cls(
            min_confidence=float(payload.get("min_confidence", 0.55)),
            max_actions_per_plan=int(payload.get("max_actions_per_plan", 8)),
            level_tolerance_lu=float(payload.get("level_tolerance_lu", 1.5)),
            max_gain_recommendation_db=float(payload.get("max_gain_recommendation_db", 3.0)),
            tonal_issue_threshold_db=float(payload.get("tonal_issue_threshold_db", 2.5)),
            max_eq_recommendation_db=float(payload.get("max_eq_recommendation_db", 2.0)),
            crest_factor_threshold_db=float(payload.get("crest_factor_threshold_db", 18.0)),
            dynamic_range_threshold_db=float(payload.get("dynamic_range_threshold_db", 18.0)),
            max_pan_recommendation=float(payload.get("max_pan_recommendation", 0.2)),
            true_peak_boost_ceiling_dbtp=float(payload.get("true_peak_boost_ceiling_dbtp", -3.0)),
        )


ACTION_ALIASES = {
    "gain": ACTION_GAIN,
    "gain_correction": ACTION_GAIN,
    "level_balance": ACTION_GAIN,
    "fader": ACTION_GAIN,
    "eq": ACTION_EQ,
    "eq_correction": ACTION_EQ,
    "spectral_shaping": ACTION_EQ,
    "compression": ACTION_COMPRESSION,
    "compression_correction": ACTION_COMPRESSION,
    "dynamics": ACTION_COMPRESSION,
    "pan": ACTION_PAN,
    "pan_correction": ACTION_PAN,
    "panning": ACTION_PAN,
    "no_action": ACTION_NO_ACTION,
}

DEFAULT_TARGET_LUFS = {
    "kick": -25.0,
    "snare": -25.0,
    "toms": -27.0,
    "overheads": -30.0,
    "bass": -23.0,
    "electric_guitar": -23.0,
    "acoustic_guitar": -25.0,
    "keys": -22.0,
    "lead_vocal": -20.0,
    "backing_vocal": -23.0,
    "master_bus": -23.0,
}

TONAL_PROBLEMS = (
    ("mud", ("MudIndex", "mud_index", "mud_db", "mud_excess_db"), 320.0, "Cut low-mid mud"),
    ("harshness", ("HarshnessIndex", "harshness_index", "harshness_db"), 3600.0, "Reduce harsh presence"),
    ("sibilance", ("SibilanceIndex", "sibilance_index", "sibilance_db"), 7800.0, "Reduce sibilance"),
    ("sub_buildup", ("SubIndex", "sub_index", "sub_excess_db"), 55.0, "Tighten sub buildup"),
)


class DecisionEngine:
    """Create explainable action plans from analyzer, critic and knowledge data."""

    def __init__(
        self,
        knowledge_base: MixingKnowledgeBase | Mapping[str, Any] | None = None,
        config: DecisionEngineConfig | Mapping[str, Any] | None = None,
    ):
        if isinstance(config, DecisionEngineConfig):
            self.config = config
        else:
            self.config = DecisionEngineConfig.from_mapping(config)

        if isinstance(knowledge_base, MixingKnowledgeBase):
            self.knowledge_base = knowledge_base
        elif isinstance(knowledge_base, Mapping):
            self.knowledge_base = MixingKnowledgeBase.from_mapping(knowledge_base)
        else:
            self.knowledge_base = MixingKnowledgeBase.load()

    def create_action_plan(
        self,
        analyzer_output: Any,
        critic_evaluations: Any = None,
        knowledge_rules: MixingKnowledgeBase | Mapping[str, Any] | None = None,
        *,
        mode: str = "live",
        plan_id: str | None = None,
    ) -> ActionPlan:
        """Build a v2 ``ActionPlan`` without touching mixer state."""
        analyzer = normalize_analyzer_output(analyzer_output)
        critics = normalize_critic_evaluations(critic_evaluations)
        knowledge = self._resolve_knowledge(knowledge_rules)
        source_modules = sorted(
            {
                str(analyzer.get("source_module", "analyzer")),
                "knowledge_base",
                "decision_engine_v2",
                *(["critic"] if critics.get("channels") or critics.get("global") else []),
            }
        )

        decisions: list[ActionDecision] = []
        for channel in analyzer.get("channels", []):
            decisions.extend(self._decisions_for_channel(channel, critics, knowledge, mode))
            if len(decisions) >= self.config.max_actions_per_plan:
                decisions = decisions[: self.config.max_actions_per_plan]
                break

        if not decisions:
            decisions.append(
                ActionDecision(
                    id="decision_engine_v2.no_action",
                    action_type=ACTION_NO_ACTION,
                    target="mix",
                    reason="Analyzer and critic evidence did not justify a safe correction.",
                    confidence=0.8,
                    risk_level=RISK_LOW,
                    source_modules=source_modules,
                    expected_audio_effect="No audible change.",
                    safe_to_apply=True,
                )
            )

        return ActionPlan(
            plan_id=plan_id or f"decision_engine_v2.{int(time.time() * 1000)}",
            mode=mode,
            decisions=decisions,
            source_modules=source_modules,
            input_summary={
                "channel_count": len(analyzer.get("channels", [])),
                "critic_channel_count": len(critics.get("channels", [])),
                "knowledge_categories": list(knowledge.categories_for_audit()),
            },
            notes=[
                "Decision Engine v2 only creates an ActionPlan; it does not send OSC.",
                "Safety Gate and Executor must be invoked explicitly.",
            ],
        )

    def _resolve_knowledge(
        self,
        knowledge_rules: MixingKnowledgeBase | Mapping[str, Any] | None,
    ) -> MixingKnowledgeBase:
        if isinstance(knowledge_rules, MixingKnowledgeBase):
            return knowledge_rules
        if isinstance(knowledge_rules, Mapping):
            return MixingKnowledgeBase.from_mapping(knowledge_rules)
        return self.knowledge_base

    def _decisions_for_channel(
        self,
        channel: Mapping[str, Any],
        critics: Mapping[str, Any],
        knowledge: MixingKnowledgeBase,
        mode: str,
    ) -> list[ActionDecision]:
        role = normalize_role(
            channel.get("role")
            or channel.get("source_role")
            or channel.get("instrument")
            or channel.get("preset")
        )
        target = _target_for_channel(channel)
        metrics = _metrics_for_channel(channel)
        critic = _critic_for_channel(channel, critics)
        allowed_actions = {
            ACTION_ALIASES.get(action, str(action))
            for action in knowledge.allowed_actions_for(role)
        }
        risky_actions = {
            ACTION_ALIASES.get(action, str(action))
            for action in knowledge.risky_actions_for(role)
        }
        source_modules = _source_modules(channel, critic)

        if not allowed_actions:
            return [
                self._decision(
                    action_type=ACTION_NO_ACTION,
                    target=target,
                    role=role,
                    reason=f"No live-safe actions are allowed for role '{role}'.",
                    confidence=0.7,
                    risk_level=RISK_LOW,
                    source_modules=source_modules,
                    expected_audio_effect="No audible change.",
                    safe_to_apply=True,
                    metadata={"role": role},
                )
            ]

        decisions: list[ActionDecision] = []
        decisions.extend(
            self._level_decisions(
                channel,
                metrics,
                role,
                target,
                allowed_actions,
                risky_actions,
                source_modules,
                critic,
            )
        )
        decisions.extend(
            self._eq_decisions(
                metrics,
                role,
                target,
                allowed_actions,
                risky_actions,
                source_modules,
                critic,
                knowledge,
            )
        )
        decisions.extend(
            self._compression_decisions(
                metrics,
                role,
                target,
                allowed_actions,
                risky_actions,
                source_modules,
                critic,
            )
        )
        decisions.extend(
            self._pan_decisions(
                metrics,
                role,
                target,
                allowed_actions,
                risky_actions,
                source_modules,
                critic,
            )
        )

        if not decisions:
            decisions.append(
                self._decision(
                    action_type=ACTION_NO_ACTION,
                    target=target,
                    role=role,
                    reason="All measured deviations are inside v2 hysteresis/tolerance.",
                    confidence=0.75,
                    risk_level=RISK_LOW,
                    source_modules=source_modules,
                    expected_audio_effect="No audible change.",
                    safe_to_apply=True,
                    metadata={"role": role, "mode": mode},
                )
            )
        return decisions

    def _level_decisions(
        self,
        channel: Mapping[str, Any],
        metrics: Mapping[str, Any],
        role: str,
        target: str,
        allowed_actions: set[str],
        risky_actions: set[str],
        source_modules: list[str],
        critic: Mapping[str, Any],
    ) -> list[ActionDecision]:
        if ACTION_GAIN not in allowed_actions:
            return []
        measured = _first_number(metrics, "lufs", "integrated_lufs", "lufs_integrated")
        if measured is None:
            return []
        target_lufs = _first_number(metrics, "target_lufs")
        if target_lufs is None:
            target_lufs = DEFAULT_TARGET_LUFS.get(role, DEFAULT_TARGET_LUFS["master_bus"])
        delta = float(target_lufs) - float(measured)
        if abs(delta) <= self.config.level_tolerance_lu:
            return []
        gain_db = max(
            -self.config.max_gain_recommendation_db,
            min(self.config.max_gain_recommendation_db, delta),
        )
        true_peak = _first_number(metrics, "true_peak_dbtp", "true_peak_db", "peak_db")
        risk = self._risk_for_action(ACTION_GAIN, role, risky_actions, critic)
        if gain_db > 0.0 and true_peak is not None and float(true_peak) > self.config.true_peak_boost_ceiling_dbtp:
            risk = _max_risk(risk, RISK_HIGH)
        confidence = _combine_confidence(channel, critic, base=0.72)
        return [
            self._decision(
                action_type=ACTION_GAIN,
                target=target,
                role=role,
                parameters={
                    "gain_db": round(gain_db, 3),
                    "measured_lufs": round(float(measured), 3),
                    "target_lufs": round(float(target_lufs), 3),
                    "channel_id": channel.get("channel_id"),
                },
                reason=(
                    f"{role} loudness is {delta:+.2f} LU from target; "
                    "bounded level balance correction proposed."
                ),
                confidence=confidence,
                risk_level=risk,
                source_modules=source_modules,
                expected_audio_effect=(
                    "Move the source closer to its level plane while preserving headroom."
                ),
                safe_to_apply=self._safe(confidence, risk, critic),
                metadata={"role": role, "metric": "lufs"},
            )
        ]

    def _eq_decisions(
        self,
        metrics: Mapping[str, Any],
        role: str,
        target: str,
        allowed_actions: set[str],
        risky_actions: set[str],
        source_modules: list[str],
        critic: Mapping[str, Any],
        knowledge: MixingKnowledgeBase,
    ) -> list[ActionDecision]:
        if ACTION_EQ not in allowed_actions:
            return []
        entry = knowledge.category_for(role)
        decisions: list[ActionDecision] = []
        for problem_name, keys, fallback_freq, label in TONAL_PROBLEMS:
            value = _first_number(metrics, *keys)
            if value is None or float(value) <= self.config.tonal_issue_threshold_db:
                continue
            gain_db = -min(self.config.max_eq_recommendation_db, max(0.5, float(value) * 0.4))
            freq_hz = _freq_for_problem(entry, problem_name, fallback_freq)
            risk = self._risk_for_action(ACTION_EQ, role, risky_actions, critic)
            confidence = _combine_confidence({"confidence": 0.68}, critic, base=0.68)
            decisions.append(
                self._decision(
                    action_type=ACTION_EQ,
                    target=target,
                    role=role,
                    parameters={
                        "band": _eq_band_for_frequency(freq_hz),
                        "frequency_hz": round(freq_hz, 3),
                        "gain_db": round(gain_db, 3),
                        "q": 1.2 if freq_hz < 1000.0 else 1.6,
                        "problem_range": problem_name,
                    },
                    reason=f"{label}: measured {problem_name} excess is {float(value):.2f} dB.",
                    confidence=confidence,
                    risk_level=risk,
                    source_modules=source_modules,
                    expected_audio_effect=(
                        "Reduce the measured spectral problem without reference matching."
                    ),
                    safe_to_apply=self._safe(confidence, risk, critic),
                    metadata={"role": role, "metric": problem_name},
                )
            )
            break
        return decisions

    def _compression_decisions(
        self,
        metrics: Mapping[str, Any],
        role: str,
        target: str,
        allowed_actions: set[str],
        risky_actions: set[str],
        source_modules: list[str],
        critic: Mapping[str, Any],
    ) -> list[ActionDecision]:
        if ACTION_COMPRESSION not in allowed_actions:
            return []
        crest = _first_number(metrics, "crest_factor_db")
        dynamic_range = _first_number(metrics, "dynamic_range_db", "plr_db")
        needs_compression = (
            crest is not None and float(crest) >= self.config.crest_factor_threshold_db
        ) or (
            dynamic_range is not None and float(dynamic_range) >= self.config.dynamic_range_threshold_db
        )
        if not needs_compression:
            return []
        metric_value = float(crest if crest is not None else dynamic_range)
        risk = self._risk_for_action(ACTION_COMPRESSION, role, risky_actions, critic)
        confidence = _combine_confidence({"confidence": 0.62}, critic, base=0.62)
        ratio = 2.0 if role in {"lead_vocal", "bass"} else 1.6
        return [
            self._decision(
                action_type=ACTION_COMPRESSION,
                target=target,
                role=role,
                parameters={
                    "threshold_db": -18.0,
                    "ratio": ratio,
                    "attack_ms": 15.0,
                    "release_ms": 160.0,
                    "makeup_db": 0.0,
                    "metric_value_db": round(metric_value, 3),
                },
                reason=f"{role} crest/dynamic range is high ({metric_value:.2f} dB).",
                confidence=confidence,
                risk_level=risk,
                source_modules=source_modules,
                expected_audio_effect="Gently reduce level jumps before any final gain move.",
                safe_to_apply=self._safe(confidence, risk, critic),
                metadata={"role": role, "metric": "crest_factor_or_dynamic_range"},
            )
        ]

    def _pan_decisions(
        self,
        metrics: Mapping[str, Any],
        role: str,
        target: str,
        allowed_actions: set[str],
        risky_actions: set[str],
        source_modules: list[str],
        critic: Mapping[str, Any],
    ) -> list[ActionDecision]:
        if ACTION_PAN not in allowed_actions:
            return []
        pan = _first_number(metrics, "recommended_pan", "pan_target", "pan_correction")
        if pan is None:
            return []
        pan_value = max(-1.0, min(1.0, float(pan)))
        if abs(pan_value) > self.config.max_pan_recommendation:
            pan_value = self.config.max_pan_recommendation if pan_value > 0 else -self.config.max_pan_recommendation
        risk = self._risk_for_action(ACTION_PAN, role, risky_actions, critic)
        confidence = _combine_confidence({"confidence": 0.6}, critic, base=0.6)
        return [
            self._decision(
                action_type=ACTION_PAN,
                target=target,
                role=role,
                parameters={"pan": round(pan_value, 3)},
                reason=f"{role} has a pan recommendation from analyzer/knowledge context.",
                confidence=confidence,
                risk_level=risk,
                source_modules=source_modules,
                expected_audio_effect="Small stereo placement adjustment, bounded for live mode.",
                safe_to_apply=self._safe(confidence, risk, critic),
                metadata={"role": role, "metric": "recommended_pan"},
            )
        ]

    def _decision(self, **payload: Any) -> ActionDecision:
        action_type = str(payload.get("action_type", ACTION_NO_ACTION))
        role = str(payload.pop("role", "unknown"))
        target = str(payload.get("target", "mix"))
        decision_id = payload.pop("id", None) or f"decision_engine_v2.{target}.{action_type}.{role}"
        return ActionDecision(id=decision_id, **payload)

    def _risk_for_action(
        self,
        action_type: str,
        role: str,
        risky_actions: set[str],
        critic: Mapping[str, Any],
    ) -> str:
        risk = RISK_MEDIUM if action_type in risky_actions else RISK_LOW
        critic_risk = str(critic.get("risk_level", "")).lower()
        if critic_risk in {RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_CRITICAL}:
            risk = _max_risk(risk, critic_risk)
        if bool(critic.get("blocked", False)):
            risk = _max_risk(risk, RISK_HIGH)
        if role == "master_bus" and action_type in {ACTION_GAIN, ACTION_COMPRESSION}:
            risk = _max_risk(risk, RISK_MEDIUM)
        return risk

    def _safe(self, confidence: float, risk_level: str, critic: Mapping[str, Any]) -> bool:
        if bool(critic.get("blocked", False)):
            return False
        if confidence < self.config.min_confidence:
            return False
        return risk_level not in {RISK_HIGH, RISK_CRITICAL}


def _target_for_channel(channel: Mapping[str, Any]) -> str:
    channel_id = channel.get("channel_id")
    if channel_id is not None:
        return f"channel:{channel_id}"
    return str(channel.get("target") or channel.get("name") or "mix")


def _metrics_for_channel(channel: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = dict(channel.get("metrics", {}) or {})
    for key, value in channel.items():
        if key in {"metrics", "name", "role", "source_role", "instrument", "preset"}:
            continue
        if key not in metrics:
            metrics[key] = value
    nested = dict(metrics.get("level", {}) or {})
    nested.update(dict(metrics.get("spectral", {}) or {}))
    nested.update(dict(metrics.get("dynamics", {}) or {}))
    nested.update(dict(metrics.get("mix_indexes", {}) or {}))
    nested.update(metrics)
    return nested


def _critic_for_channel(channel: Mapping[str, Any], critics: Mapping[str, Any]) -> Dict[str, Any]:
    channel_id = channel.get("channel_id")
    target = _target_for_channel(channel)
    role = normalize_role(channel.get("role") or channel.get("source_role") or channel.get("instrument"))
    for item in critics.get("channels", []):
        if channel_id is not None and item.get("channel_id") == channel_id:
            return dict(item)
        if str(item.get("target", "")) == target:
            return dict(item)
        if normalize_role(item.get("role")) == role and role != "unknown":
            return dict(item)
    return dict(critics.get("global", {}))


def _source_modules(channel: Mapping[str, Any], critic: Mapping[str, Any]) -> list[str]:
    modules = {
        "decision_engine_v2",
        "knowledge_base",
        str(channel.get("source_module", "analyzer")),
    }
    if critic:
        modules.add(str(critic.get("source_module", "critic")))
    return sorted(modules)


def _first_number(metrics: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _combine_confidence(channel: Mapping[str, Any], critic: Mapping[str, Any], *, base: float) -> float:
    analyzer_conf = _first_number(channel, "confidence", "classification_confidence")
    critic_conf = _first_number(critic, "confidence")
    values = [base]
    if analyzer_conf is not None:
        values.append(float(analyzer_conf))
    if critic_conf is not None:
        values.append(float(critic_conf))
    return clamp_confidence(sum(values) / len(values))


def _freq_for_problem(entry: Mapping[str, Any], problem_name: str, fallback: float) -> float:
    ranges = entry.get("common_problem_ranges", {})
    value = dict(ranges or {}).get(problem_name)
    if isinstance(value, Mapping):
        bounds = value.get("range_hz", [])
    else:
        bounds = value or []
    if isinstance(bounds, Sequence) and len(bounds) >= 2:
        try:
            low = float(bounds[0])
            high = float(bounds[1])
            return (low * high) ** 0.5
        except (TypeError, ValueError):
            return fallback
    return fallback


def _eq_band_for_frequency(freq_hz: float) -> int:
    if freq_hz < 180.0:
        return 1
    if freq_hz < 1200.0:
        return 2
    if freq_hz < 5500.0:
        return 3
    return 4


RISK_ORDER = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2, RISK_CRITICAL: 3}


def _max_risk(left: str, right: str) -> str:
    left_key = left if left in RISK_ORDER else RISK_MEDIUM
    right_key = right if right in RISK_ORDER else RISK_MEDIUM
    return left_key if RISK_ORDER[left_key] >= RISK_ORDER[right_key] else right_key
