"""EWMA drift guards for MuQ-style quality observations.

The drift sign is intentionally asymmetric: a lower MuQ quality score than the
baseline is treated as positive drift because it is the live-sound risk case.
Quality improvements do not trigger WARN/CRIT states.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Any, Dict, Optional, Tuple


class DriftState(str, Enum):
    """Debounced quality drift state."""

    NORMAL = "NORMAL"
    WARN = "WARN"
    CRIT = "CRIT"


DRIFT_STATE_ORDER = {
    DriftState.NORMAL: 0,
    DriftState.WARN: 1,
    DriftState.CRIT: 2,
}


def _finite_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _deepcopy_jsonish(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _deepcopy_jsonish(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_deepcopy_jsonish(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return deepcopy(value)


@dataclass
class EwmaDrift:
    """Stateful EWMA drift detector for a single MuQ score stream."""

    ewma: Optional[float] = None
    baseline: Optional[float] = None
    tau: float = 12.0
    warn_T: float = 0.06
    crit_T: float = 0.12
    debounce_warn: float = 2.0
    debounce_crit: float = 5.0
    warn_timer: float = 0.0
    crit_timer: float = 0.0
    last_known_good: Optional[Dict[str, Any]] = None
    _state: DriftState = field(default=DriftState.NORMAL, init=False, repr=False)

    @property
    def state(self) -> DriftState:
        """Current debounced state."""

        return self._state

    def set_baseline(self, baseline: Optional[float] = None) -> float:
        """Set the drift baseline and clear debounce timers.

        Args:
            baseline: Explicit normalized MuQ baseline. When omitted, the
                current EWMA is used, falling back to 0.0.

        Returns:
            The baseline stored by the detector.
        """

        resolved = _finite_float(baseline, self.ewma if self.ewma is not None else 0.0)
        self.baseline = float(resolved if resolved is not None else 0.0)
        self.warn_timer = 0.0
        self.crit_timer = 0.0
        self._state = DriftState.NORMAL
        return self.baseline

    def snapshot_params(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        """Store a last-known-good parameter snapshot for later restore."""

        self.last_known_good = _deepcopy_jsonish(dict(params))
        return deepcopy(self.last_known_good)

    def restore_last_good(self) -> Optional[Dict[str, Any]]:
        """Return a copy of the latest known-good parameter snapshot."""

        if self.last_known_good is None:
            return None
        return deepcopy(self.last_known_good)

    def update(self, muq_obs: float, dt: float) -> Dict[str, Any]:
        """Update EWMA and return drift state and requested actions.

        Args:
            muq_obs: Normalized MuQ observation where higher is better.
            dt: Seconds since the previous observation.

        Returns:
            Dict with `ewma`, `drift`, `state`, `actions`, timers, and baseline.
        """

        obs = _finite_float(muq_obs)
        if obs is None:
            return self._result(
                drift=self._current_drift(),
                actions=["ignore_invalid_observation"],
                observed=None,
            )

        dt = max(0.0, _finite_float(dt, 0.0) or 0.0)
        tau = max(1e-6, _finite_float(self.tau, 12.0) or 12.0)

        if self.ewma is None:
            self.ewma = obs
        else:
            alpha = 1.0 - math.exp(-dt / tau)
            self.ewma += alpha * (obs - self.ewma)

        if self.baseline is None:
            self.baseline = float(self.ewma)

        drift = self._current_drift()
        previous_state = self._state
        next_state = self._advance_state(drift, dt)
        actions = self._actions_for_transition(previous_state, next_state)
        self._state = next_state

        return self._result(drift=drift, actions=actions, observed=obs)

    def _current_drift(self) -> float:
        if self.ewma is None or self.baseline is None:
            return 0.0
        return float(max(0.0, self.baseline - self.ewma))

    def _advance_state(self, drift: float, dt: float) -> DriftState:
        warn_release = max(0.0, float(self.warn_T) * 0.75)
        crit_release = max(warn_release, float(self.crit_T) * 0.75)

        warn_threshold = warn_release if self._state != DriftState.NORMAL else float(self.warn_T)
        crit_threshold = crit_release if self._state == DriftState.CRIT else float(self.crit_T)

        if drift >= warn_threshold:
            self.warn_timer += dt
        elif drift < warn_release:
            self.warn_timer = 0.0

        if drift >= crit_threshold:
            self.crit_timer += dt
        elif drift < crit_release:
            self.crit_timer = 0.0

        if self._state == DriftState.CRIT and drift >= crit_release:
            return DriftState.CRIT
        if self.crit_timer >= max(0.0, float(self.debounce_crit)):
            return DriftState.CRIT
        if self._state == DriftState.WARN and drift >= warn_release:
            return DriftState.WARN
        if self.warn_timer >= max(0.0, float(self.debounce_warn)):
            return DriftState.WARN
        return DriftState.NORMAL

    @staticmethod
    def _actions_for_transition(
        previous_state: DriftState,
        next_state: DriftState,
    ) -> list[str]:
        if next_state == DriftState.CRIT:
            actions = ["log_crit", "osc_highlight_crit", "freeze_ml_corrections"]
            if previous_state != DriftState.CRIT:
                actions.append("restore_last_good")
            return actions
        if next_state == DriftState.WARN:
            return ["increase_logging", "osc_highlight_warn"]
        if previous_state != DriftState.NORMAL:
            return ["clear_osc_highlight"]
        return []

    def _result(
        self,
        *,
        drift: float,
        actions: list[str],
        observed: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "ewma": float(self.ewma) if self.ewma is not None else None,
            "baseline": float(self.baseline) if self.baseline is not None else None,
            "drift": float(drift),
            "state": self._state.value,
            "actions": list(actions),
            "warn_timer": float(self.warn_timer),
            "crit_timer": float(self.crit_timer),
            "observed": float(observed) if observed is not None else None,
        }


DEFAULT_DRIFT_PROFILE: Dict[str, float] = {
    "tau": 12.0,
    "warn_T": 0.06,
    "crit_T": 0.12,
    "debounce_warn": 2.0,
    "debounce_crit": 5.0,
}

DEFAULT_GROUP_PROFILES: Dict[str, Dict[str, float]] = {
    "vox": {"tau": 9.0, "warn_T": 0.05, "crit_T": 0.10, "debounce_warn": 1.5},
    "drums": {"tau": 7.0, "warn_T": 0.07, "crit_T": 0.14, "debounce_warn": 1.2},
    "bass": {"tau": 10.0, "warn_T": 0.06, "crit_T": 0.12},
    "guitars": {"tau": 12.0, "warn_T": 0.06, "crit_T": 0.12},
    "keys": {"tau": 14.0, "warn_T": 0.06, "crit_T": 0.12},
    "fx": {"tau": 18.0, "warn_T": 0.08, "crit_T": 0.16, "debounce_warn": 2.5},
    "bus": {"tau": 15.0, "warn_T": 0.05, "crit_T": 0.10, "debounce_crit": 4.0},
}

DEFAULT_STEM_ALIASES: Dict[str, Tuple[str, ...]] = {
    "vox": ("vox", "vocal", "lead", "backing", "bgv", "choir"),
    "drums": ("drum", "kick", "snare", "tom", "cymbal", "overhead", "room"),
    "bass": ("bass", "sub"),
    "guitars": ("guitar", "gtr"),
    "keys": ("keys", "keyboard", "piano", "synth", "organ"),
    "fx": ("fx", "effect", "reverb", "delay", "return"),
    "bus": ("bus", "master", "main", "mix"),
}


def default_ewma_metrics_config() -> Dict[str, Any]:
    """Return the disabled-by-default EWMA metrics config."""

    return {
        "enabled": False,
        "freeze_normal_seconds": 5.0,
        "default": dict(DEFAULT_DRIFT_PROFILE),
        "groups": deepcopy(DEFAULT_GROUP_PROFILES),
        "stem_aliases": {key: list(value) for key, value in DEFAULT_STEM_ALIASES.items()},
        "frequency_masks": {
            "vox": {
                "low_mid_300_800": {
                    "band_hz": [300, 800],
                    "tau": 7.0,
                    "warn_T": 0.04,
                    "crit_T": 0.08,
                    "debounce_warn": 1.2,
                    "debounce_crit": 3.0,
                }
            },
            "drums": {
                "body_120_400": {
                    "band_hz": [120, 400],
                    "tau": 5.0,
                    "warn_T": 0.06,
                    "crit_T": 0.12,
                }
            },
            "bus": {
                "presence_2500_6000": {
                    "band_hz": [2500, 6000],
                    "tau": 10.0,
                    "warn_T": 0.04,
                    "crit_T": 0.09,
                }
            },
        },
        "visualization": {
            "osc_endpoint": "/autofoh/muq_drift",
            "emit_observation": True,
        },
    }


def _profile_from_mapping(config: Mapping[str, Any], group: str) -> Dict[str, float]:
    profile = dict(DEFAULT_DRIFT_PROFILE)
    default_profile = config.get("default", {})
    if isinstance(default_profile, Mapping):
        profile.update(
            {
                key: value
                for key, value in default_profile.items()
                if key in DEFAULT_DRIFT_PROFILE
            }
        )
    groups = config.get("groups", {})
    if isinstance(groups, Mapping):
        group_profile = groups.get(group, {})
        if isinstance(group_profile, Mapping):
            profile.update(
                {
                    key: value
                    for key, value in group_profile.items()
                    if key in DEFAULT_DRIFT_PROFILE
                }
            )
    return {key: float(value) for key, value in profile.items()}


def _extract_score(observation: Any) -> Optional[float]:
    if isinstance(observation, Mapping):
        for key in ("muq_score", "quality_score", "score", "value", "MI", "mi"):
            if key in observation:
                return _finite_float(observation[key])
        return None
    return _finite_float(observation)


def _extract_band_scores(observation: Any) -> Dict[str, float]:
    if not isinstance(observation, Mapping):
        return {}
    for key in ("bands", "masks", "frequency_masks", "band_scores"):
        value = observation.get(key)
        if not isinstance(value, Mapping):
            continue
        scores: Dict[str, float] = {}
        for band_name, band_value in value.items():
            score = _extract_score(band_value)
            if score is not None:
                scores[str(band_name)] = score
        return scores
    return {}


def _merge_enabled_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    merged = default_ewma_metrics_config()
    if not isinstance(config, Mapping):
        return merged

    payload = config
    if isinstance(config.get("ewma_metrics"), Mapping):
        payload = config["ewma_metrics"]
    if isinstance(config.get("stem_drift"), Mapping):
        payload = config["stem_drift"]

    for key, value in payload.items():
        if key in {"default", "groups", "stem_aliases", "frequency_masks", "visualization"}:
            if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
                merged[key] = _deep_merge_dict(merged[key], value)
            else:
                merged[key] = deepcopy(value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class StemEwmaDriftMonitor:
    """Per-stem and optional per-mask EWMA drift monitor."""

    FULL_BAND = "full_band"

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = _merge_enabled_config(config)
        self.enabled = bool(self.config.get("enabled", False))
        self.freeze_normal_seconds = float(self.config.get("freeze_normal_seconds", 5.0))
        self.trackers: Dict[Tuple[str, str], EwmaDrift] = {}
        self._frozen_stems: Dict[str, Dict[str, Any]] = {}
        self._last_update_at: Optional[float] = None

    def update_batch(
        self,
        stem_scores: Mapping[str, Any],
        dt: Optional[float] = None,
        *,
        params_by_stem: Optional[Mapping[str, Mapping[str, Any]]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update all stem trackers from a MuQ batch."""

        if not self.enabled:
            return {
                "enabled": False,
                "timestamp": float(time.time() if timestamp is None else timestamp),
                "stems": {},
                "frozen_stems": {},
                "summary": "MuQ stem EWMA drift disabled",
            }
        if not isinstance(stem_scores, Mapping):
            raise TypeError("stem_scores must be a mapping of stem name to MuQ observation")

        now = float(time.time() if timestamp is None else timestamp)
        if dt is None:
            dt = 0.0 if self._last_update_at is None else now - self._last_update_at
        dt = max(0.0, float(dt))
        self._last_update_at = now

        params_by_stem = params_by_stem or {}
        stem_results: Dict[str, Any] = {}
        for stem_name, observation in stem_scores.items():
            stem = str(stem_name)
            group = self._group_for_observation(stem, observation)
            params = params_by_stem.get(stem) or {}
            result = self._update_stem(
                stem=stem,
                group=group,
                observation=observation,
                dt=dt,
                params=params,
            )
            if result:
                stem_results[stem] = result

        self._advance_freeze_recovery(stem_results, dt)
        for stem, result in stem_results.items():
            result["frozen"] = self.is_stem_frozen(stem)
        return {
            "enabled": True,
            "timestamp": now,
            "dt": dt,
            "stems": stem_results,
            "frozen_stems": self.frozen_stems(),
            "summary": self.summary(stem_results),
        }

    def frozen_stems(self) -> Dict[str, Dict[str, Any]]:
        """Return currently frozen stems and their recovery timers."""

        return {
            stem: deepcopy(info)
            for stem, info in self._frozen_stems.items()
            if bool(info.get("frozen", False))
        }

    def is_stem_frozen(self, stem_name: str) -> bool:
        info = self._frozen_stems.get(str(stem_name), {})
        return bool(info.get("frozen", False))

    def restore_stem_last_good(self, stem_name: str) -> Optional[Dict[str, Any]]:
        """Return the full-band last-known-good snapshot for a stem."""

        tracker = self.trackers.get((str(stem_name), self.FULL_BAND))
        if tracker is None:
            return None
        return tracker.restore_last_good()

    def _update_stem(
        self,
        *,
        stem: str,
        group: str,
        observation: Any,
        dt: float,
        params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        masks: Dict[str, Any] = {}
        full_score = _extract_score(observation)
        if full_score is not None:
            masks[self.FULL_BAND] = full_score
        masks.update(_extract_band_scores(observation))

        if not masks:
            return {}

        entries: Dict[str, Any] = {}
        for mask_name, score in masks.items():
            tracker = self._tracker_for(stem, group, mask_name)
            result = tracker.update(score, dt)
            if result["state"] == DriftState.NORMAL.value and params:
                tracker.snapshot_params(params)
            if "restore_last_good" in result["actions"]:
                result["restored_params"] = tracker.restore_last_good()
            entries[mask_name] = result

        worst_state = self._worst_state(result["state"] for result in entries.values())
        actions = self._aggregate_actions(entries.values())
        return {
            "group": group,
            "state": worst_state.value,
            "actions": actions,
            "masks": entries,
            "frozen": self.is_stem_frozen(stem),
            "osc_endpoint": self._osc_endpoint(stem),
        }

    def _tracker_for(self, stem: str, group: str, mask_name: str) -> EwmaDrift:
        key = (stem, mask_name)
        tracker = self.trackers.get(key)
        if tracker is not None:
            return tracker

        profile = _profile_from_mapping(self.config, group)
        if mask_name != self.FULL_BAND:
            mask_profile = self._mask_profile(group, mask_name)
            profile.update(mask_profile)
        tracker = EwmaDrift(**profile)
        self.trackers[key] = tracker
        return tracker

    def _mask_profile(self, group: str, mask_name: str) -> Dict[str, float]:
        masks = self.config.get("frequency_masks", {})
        if not isinstance(masks, Mapping):
            return {}
        group_masks = masks.get(group, {})
        if not isinstance(group_masks, Mapping):
            return {}
        profile = group_masks.get(mask_name, {})
        if not isinstance(profile, Mapping):
            return {}
        return {
            key: float(value)
            for key, value in profile.items()
            if key in DEFAULT_DRIFT_PROFILE
        }

    def _group_for_observation(self, stem: str, observation: Any) -> str:
        if isinstance(observation, Mapping):
            group = str(observation.get("group", "")).strip().lower()
            if group:
                return group
        return self.infer_group(stem)

    def infer_group(self, stem: str) -> str:
        normalized = str(stem).lower().replace("-", "_").replace(" ", "_")
        aliases = self.config.get("stem_aliases", DEFAULT_STEM_ALIASES)
        if not isinstance(aliases, Mapping):
            aliases = DEFAULT_STEM_ALIASES
        for group, values in aliases.items():
            if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                values = (str(values),)
            if any(str(alias).lower() in normalized for alias in values):
                return str(group)
        return "default"

    def _advance_freeze_recovery(self, stem_results: Mapping[str, Any], dt: float) -> None:
        for stem, result in stem_results.items():
            state = DriftState(result.get("state", DriftState.NORMAL.value))
            actions = set(result.get("actions", []))
            info = self._frozen_stems.setdefault(
                stem,
                {"frozen": False, "normal_timer": 0.0, "since": None},
            )
            if state == DriftState.CRIT or "freeze_ml_corrections" in actions:
                if not info.get("frozen", False):
                    info["since"] = time.time()
                info["frozen"] = True
                info["normal_timer"] = 0.0
                continue
            if not info.get("frozen", False):
                continue
            if state == DriftState.NORMAL:
                info["normal_timer"] = float(info.get("normal_timer", 0.0)) + dt
                if info["normal_timer"] >= self.freeze_normal_seconds:
                    info["frozen"] = False
                    info["released_at"] = time.time()
            else:
                info["normal_timer"] = 0.0

    @staticmethod
    def _worst_state(states: Iterable[str]) -> DriftState:
        worst = DriftState.NORMAL
        for state in states:
            candidate = DriftState(state)
            if DRIFT_STATE_ORDER[candidate] > DRIFT_STATE_ORDER[worst]:
                worst = candidate
        return worst

    @staticmethod
    def _aggregate_actions(entries: Iterable[Mapping[str, Any]]) -> list[str]:
        actions: list[str] = []
        for entry in entries:
            for action in entry.get("actions", []):
                if action not in actions:
                    actions.append(str(action))
        return actions

    def _osc_endpoint(self, stem: str) -> str:
        visualization = self.config.get("visualization", {})
        endpoint = "/autofoh/muq_drift"
        if isinstance(visualization, Mapping):
            endpoint = str(visualization.get("osc_endpoint", endpoint))
        return f"{endpoint}/{stem}"

    @staticmethod
    def summary(stem_results: Mapping[str, Any]) -> str:
        if not stem_results:
            return "MuQ stem EWMA drift: no valid stem scores"
        counts = {state.value: 0 for state in DriftState}
        for result in stem_results.values():
            state = str(result.get("state", DriftState.NORMAL.value))
            counts[state] = counts.get(state, 0) + 1
        return (
            "MuQ stem EWMA drift: "
            f"NORMAL={counts.get('NORMAL', 0)} "
            f"WARN={counts.get('WARN', 0)} "
            f"CRIT={counts.get('CRIT', 0)}"
        )
