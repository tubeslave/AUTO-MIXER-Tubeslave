from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class EvaluatorAgent:
    def __init__(
        self,
        name: str = "evaluator",
        project_root: str | Path = ".",
        runs_dir: str | Path = "artifacts/runs",
    ) -> None:
        self.name = name
        self.project_root = Path(project_root).resolve()
        self.runs_dir = (self.project_root / runs_dir).resolve()

    def _confined_run_path(self, raw_path: Any) -> Path:
        path = Path(str(raw_path)).resolve()
        try:
            path.relative_to(self.runs_dir)
        except ValueError as exc:
            raise ValueError("evaluation path must remain beneath the configured runs directory") from exc
        return path

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

        try:
            model_path = self._confined_run_path(model_dir)
        except ValueError as exc:
            return {"agent": self.name, "status": "failed", "error": str(exc)}
        if not model_path.is_dir():
            return {
                "agent": self.name,
                "status": "failed",
                "error": "model_dir does not exist or is not a directory",
                "result": {
                    "model_dir": str(model_path),
                    "model_dir_exists": False,
                },
            }

        artifact_paths = sorted(
            path
            for pattern in ("evaluation.json", "training_summary.json", "metadata.json")
            for path in model_path.rglob(pattern)
            if path.is_file()
        )[:256]
        parsed_artifacts = []
        invalid_artifacts = []
        for artifact_path in artifact_paths:
            try:
                if artifact_path.is_symlink() or artifact_path.stat().st_size > 1_000_000:
                    raise ValueError("artifact is a symlink or exceeds the byte limit")
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                invalid_artifacts.append({
                    "path": str(artifact_path),
                    "error": type(exc).__name__,
                })
                continue
            parsed_artifacts.append({
                "path": str(artifact_path),
                "kind": artifact_path.name,
                "metrics": payload.get("metrics", {}) if isinstance(payload, dict) else {},
            })

        status = "ok" if parsed_artifacts else "incomplete"
        return {
            "agent": self.name,
            "status": status,
            "result": {
                "model_dir": str(model_path),
                "model_dir_exists": True,
                "artifact_count": len(parsed_artifacts),
                "invalid_artifact_count": len(invalid_artifacts),
                "artifacts": parsed_artifacts,
                "invalid_artifacts": invalid_artifacts,
                "summary": (
                    "Existing evaluation artifacts parsed"
                    if parsed_artifacts
                    else "No parseable evaluation artifacts found"
                ),
                "listening_test_completed": False,
                "live_authority_granted": False,
            },
        }

    def _evaluate_best_run(self, best_run: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(best_run, dict):
            return {
                "agent": self.name,
                "status": "failed",
                "error": "best_run must be a mapping",
            }
        missing_fields = [
            field
            for field in ("run_id", "run_dir", "log_path")
            if not best_run.get(field)
        ]
        if missing_fields:
            return {
                "agent": self.name,
                "status": "failed",
                "error": f"best_run missing required fields: {', '.join(missing_fields)}",
            }

        try:
            run_dir = self._confined_run_path(best_run["run_dir"])
            log_path = Path(best_run["log_path"]).resolve()
            log_path.relative_to(run_dir)
        except ValueError as exc:
            return {"agent": self.name, "status": "failed", "error": str(exc)}
        run_dir_exists = run_dir.is_dir()
        log_exists = log_path.is_file()
        training_success = best_run.get("success") is True
        if not run_dir_exists:
            return {
                "agent": self.name,
                "status": "failed",
                "error": "best_run run_dir does not exist or is not a directory",
                "evaluated_run": best_run["run_id"],
                "checks": {
                    "run_dir_exists": False,
                    "log_exists": log_exists,
                    "training_success": training_success,
                },
            }

        evaluation = {
            "agent": self.name,
            "status": "ok" if log_exists and training_success else "failed",
            "evaluated_run": best_run["run_id"],
            "metrics": best_run.get("metrics", {}),
            "checks": {
                "run_dir_exists": run_dir_exists,
                "log_exists": log_exists,
                "training_success": training_success,
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
