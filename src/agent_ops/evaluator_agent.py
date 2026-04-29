from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluatorAgent:
    def __init__(self, name: str = "evaluator") -> None:
        self.name = name

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        best_run = data.get("best_run")
        if best_run:
            return self._evaluate_best_run(best_run)

        model_dir = data.get("model_dir")
        if model_dir is None:
            return {
                "agent": self.name,
                "status": "failed",
                "error": "No best_run provided",
            }

        exists = bool(model_dir and Path(model_dir).exists())
        return {
            "agent": self.name,
            "status": "ok",
            "result": {
                "model_dir": model_dir,
                "model_dir_exists": exists,
                "summary": "Basic evaluation placeholder",
                "next_recommendation": "Attach metrics parser and listening-test pipeline.",
            },
        }

    def _evaluate_best_run(self, best_run: dict[str, Any]) -> dict[str, Any]:
        run_dir = Path(best_run["run_dir"])
        log_path = Path(best_run["log_path"])
        evaluation = {
            "agent": self.name,
            "status": "ok",
            "evaluated_run": best_run["run_id"],
            "metrics": best_run.get("metrics", {}),
            "checks": {
                "run_dir_exists": run_dir.exists(),
                "log_exists": log_path.exists(),
                "training_success": best_run.get("success", False),
            },
            "recommendation": (
                "Если loss стабильно падает — добавь validation set и perceptual metrics."
            ),
        }
        eval_path = run_dir / "evaluation.json"
        eval_path.write_text(
            json.dumps(evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        evaluation["evaluation_path"] = str(eval_path)
        return evaluation
