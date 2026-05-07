"""Optional Nevergrad action optimizer adapter."""

from __future__ import annotations

from typing import Any

from .action_schema import CandidateActionSet, EQAction, GainAction
from .optimizer_base import OptimizerStatus


class NevergradActionOptimizer:
    """Ask/tell optimizer that proposes bounded action sets only."""

    def __init__(self, config: dict[str, Any] | None, action_space: dict[str, Any], random_seed: int | None = None):
        self.config = dict(config or {})
        self.action_space = dict(action_space or {})
        self._asked: dict[str, CandidateActionSet] = {}
        self._ng_candidates: dict[str, Any] = {}
        self._results: dict[str, float] = {}
        try:
            import nevergrad as ng  # type: ignore

            self._ng = ng
            budget = int(self.config.get("budget", self.config.get("max_candidates", 20)) or 20)
            self._parametrization = self._build_parametrization()
            self._optimizer = ng.optimizers.OnePlusOne(
                parametrization=self._parametrization,
                budget=max(1, budget),
                num_workers=1,
            )
            if random_seed is not None:
                self._optimizer.parametrization.random_state.seed(int(random_seed))
            self.status = OptimizerStatus("nevergrad", True)
        except Exception as exc:
            self._ng = None
            self._parametrization = None
            self._optimizer = None
            self.status = OptimizerStatus(
                "nevergrad",
                False,
                [f"nevergrad unavailable; manual candidates/fallback will be used: {exc}"],
            )

    def ask_candidates(self, n_candidates: int) -> list[CandidateActionSet]:
        if self._ng is None or self._optimizer is None:
            return []
        candidates: list[CandidateActionSet] = []
        for index in range(max(0, int(n_candidates))):
            ng_candidate = self._optimizer.ask()
            params = self._coerce_params(ng_candidate.value)
            actions = self._params_to_actions(params)
            candidate = CandidateActionSet(
                candidate_id=f"nevergrad_{index:03d}",
                actions=actions,
                description="Nevergrad bounded parameter proposal.",
                source="nevergrad",
                safety_limits_snapshot=self.action_space.get("safety", {}),
                metadata={"parameters": params},
            )
            self._asked[candidate.candidate_id] = candidate
            self._ng_candidates[candidate.candidate_id] = ng_candidate
            candidates.append(candidate)
        return candidates

    def tell_result(self, candidate_id: str, score: float, metadata: dict[str, Any]) -> None:
        self._results[candidate_id] = float(score)
        self.status.history.append({"candidate_id": candidate_id, "score": float(score), "metadata": dict(metadata)})
        if self._optimizer is not None and candidate_id in self._ng_candidates:
            self._optimizer.tell(self._ng_candidates[candidate_id], -float(score))

    def recommend_best(self) -> CandidateActionSet | None:
        if not self._results:
            return None
        best_id = max(self._results.items(), key=lambda item: item[1])[0]
        return self._asked.get(best_id)

    def _build_parametrization(self) -> Any:
        assert self._ng is not None
        return self._ng.p.Dict(
            vocal_gain_db=self._ng.p.Scalar(lower=-1.0, upper=1.0),
            bass_gain_db=self._ng.p.Scalar(lower=-1.0, upper=1.0),
            kick_gain_db=self._ng.p.Scalar(lower=-1.0, upper=1.0),
            snare_gain_db=self._ng.p.Scalar(lower=-1.0, upper=1.0),
            guitars_gain_db=self._ng.p.Scalar(lower=-1.0, upper=1.0),
            low_mid_cut_db=self._ng.p.Scalar(lower=-2.0, upper=0.0),
            harshness_cut_db=self._ng.p.Scalar(lower=-2.0, upper=0.0),
            master_trim_db=self._ng.p.Scalar(lower=-1.0, upper=0.5),
        )

    @staticmethod
    def _coerce_params(value: Any) -> dict[str, float]:
        payload = dict(value or {})
        return {
            "vocal_gain_db": float(payload.get("vocal_gain_db", 0.0)),
            "bass_gain_db": float(payload.get("bass_gain_db", 0.0)),
            "kick_gain_db": float(payload.get("kick_gain_db", 0.0)),
            "snare_gain_db": float(payload.get("snare_gain_db", 0.0)),
            "guitars_gain_db": float(payload.get("guitars_gain_db", 0.0)),
            "low_mid_cut_db": float(payload.get("low_mid_cut_db", 0.0)),
            "harshness_cut_db": float(payload.get("harshness_cut_db", 0.0)),
            "master_trim_db": float(payload.get("master_trim_db", 0.0)),
        }

    def _params_to_actions(self, params: dict[str, float]) -> list[Any]:
        channels = self.action_space.get("channels", {}) or {}
        actions: list[Any] = []
        role_to_param = {
            "vocal": "vocal_gain_db",
            "bass": "bass_gain_db",
            "kick": "kick_gain_db",
            "snare": "snare_gain_db",
            "guitars": "guitars_gain_db",
        }
        for channel_id, role in channels.items():
            for role_name, param_name in role_to_param.items():
                if role_name in role:
                    value = float(params[param_name])
                    if abs(value) >= 0.05:
                        actions.append(GainAction(channel_id, round(value, 3)))
                    break
            if role not in {"kick", "bass"} and params["low_mid_cut_db"] < -0.2:
                actions.append(EQAction(channel_id, "low_mid", 250.0, round(params["low_mid_cut_db"], 3), 0.9, "peaking"))
            if role in {"vocal", "guitars", "snare", "drums"} and params["harshness_cut_db"] < -0.2:
                actions.append(EQAction(channel_id, "harshness", 3500.0, round(params["harshness_cut_db"], 3), 1.2, "peaking"))
        if abs(params["master_trim_db"]) >= 0.05:
            actions.append(GainAction("master", round(params["master_trim_db"], 3)))
        return actions
