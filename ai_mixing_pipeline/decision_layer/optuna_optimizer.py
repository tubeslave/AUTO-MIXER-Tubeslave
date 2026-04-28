"""Optional Optuna optimizer adapter for offline batch experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .action_schema import CandidateActionSet, EQAction, GainAction
from .optimizer_base import OptimizerStatus


class OptunaActionOptimizer:
    """Experimental define-by-run style optimizer with graceful fallback."""

    def __init__(self, config: dict[str, Any] | None, action_space: dict[str, Any], random_seed: int | None = None):
        self.config = dict(config or {})
        self.action_space = dict(action_space or {})
        self._asked: dict[str, CandidateActionSet] = {}
        self._trials: dict[str, Any] = {}
        self._results: dict[str, float] = {}
        try:
            import optuna  # type: ignore

            self._optuna = optuna
            sampler = optuna.samplers.TPESampler(seed=random_seed)
            self._study = optuna.create_study(direction="maximize", sampler=sampler)
            self.status = OptimizerStatus("optuna", True)
        except Exception as exc:
            self._optuna = None
            self._study = None
            self.status = OptimizerStatus(
                "optuna",
                False,
                [f"optuna unavailable; optimizer candidates skipped: {exc}"],
            )

    def ask_candidates(self, n_candidates: int) -> list[CandidateActionSet]:
        if self._optuna is None or self._study is None:
            return []
        candidates: list[CandidateActionSet] = []
        for index in range(max(0, int(n_candidates))):
            trial = self._study.ask()
            params = {
                "vocal_gain_db": trial.suggest_float("vocal_gain_db", -1.0, 1.0),
                "low_mid_cut_db": trial.suggest_float("low_mid_cut_db", -2.0, 0.0),
                "master_trim_db": trial.suggest_float("master_trim_db", -1.0, 0.5),
            }
            actions = self._params_to_actions(params)
            candidate = CandidateActionSet(
                candidate_id=f"optuna_{index:03d}",
                actions=actions,
                description="Optuna bounded trial proposal.",
                source="optuna",
                safety_limits_snapshot=self.action_space.get("safety", {}),
                metadata={"parameters": params},
            )
            self._asked[candidate.candidate_id] = candidate
            self._trials[candidate.candidate_id] = trial
            candidates.append(candidate)
        return candidates

    def tell_result(self, candidate_id: str, score: float, metadata: dict[str, Any]) -> None:
        self._results[candidate_id] = float(score)
        self.status.history.append({"candidate_id": candidate_id, "score": float(score), "metadata": dict(metadata)})
        if self._study is not None and candidate_id in self._trials:
            self._study.tell(self._trials[candidate_id], float(score))

    def recommend_best(self) -> CandidateActionSet | None:
        if not self._results:
            return None
        best_id = max(self._results.items(), key=lambda item: item[1])[0]
        return self._asked.get(best_id)

    def save_history(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.status.to_dict(), indent=2), encoding="utf-8")
        return target

    def _params_to_actions(self, params: dict[str, float]) -> list[Any]:
        channels = self.action_space.get("channels", {}) or {}
        actions: list[Any] = []
        for channel_id, role in channels.items():
            if "vocal" in role and abs(params["vocal_gain_db"]) >= 0.05:
                actions.append(GainAction(channel_id, round(params["vocal_gain_db"], 3)))
            if role not in {"kick", "bass"} and params["low_mid_cut_db"] < -0.2:
                actions.append(EQAction(channel_id, "low_mid", 250.0, round(params["low_mid_cut_db"], 3), 0.9, "peaking"))
        if abs(params["master_trim_db"]) >= 0.05:
            actions.append(GainAction("master", round(params["master_trim_db"], 3)))
        return actions
