from __future__ import annotations

from pathlib import Path
from typing import Any


class ArchitectAgent:
    def __init__(
        self,
        name: str = "architect",
        project_root: str | Path = ".",
    ) -> None:
        self.name = name
        self.project_root = Path(project_root).resolve()

    def run(self, project_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(project_state, dict):
            return {
                "agent": self.name,
                "status": "failed",
                "error": "project_state must be a mapping",
            }

        checks: dict[str, bool] = {}
        issues: list[str] = []
        task = str(project_state.get("task", "train_pipeline"))

        if task == "discover_datasets":
            discovery = project_state.get("dataset_discovery")
            checks["dataset_discovery_config_present"] = isinstance(discovery, dict)
            if isinstance(discovery, dict):
                output_dir = discovery.get("output_dir")
                checks["output_dir_declared"] = bool(output_dir)
                checks["bounded_download_size"] = (
                    isinstance(discovery.get("max_dataset_bytes"), int)
                    and discovery["max_dataset_bytes"] > 0
                )
            else:
                checks["output_dir_declared"] = False
                checks["bounded_download_size"] = False
        else:
            train_config = project_state.get("train_config")
            checks["train_config_present"] = isinstance(train_config, dict)
            script_path = train_config.get("script_path") if isinstance(train_config, dict) else None
            checks["training_entrypoint_declared"] = bool(script_path)
            checks["training_entrypoint_exists"] = bool(
                script_path and (self.project_root / str(script_path)).is_file()
            )

        model_dir = project_state.get("model_dir")
        checks["model_dir_declared"] = bool(model_dir)
        if model_dir:
            checks["model_dir_parent_exists"] = (
                self.project_root / str(model_dir)
            ).parent.exists()

        for name, passed in checks.items():
            if not passed:
                issues.append(name)

        return {
            "agent": self.name,
            "status": "ok" if not issues else "incomplete",
            "result": {
                "task": task,
                "checks": checks,
                "issues": issues,
                "summary": (
                    "Offline pipeline preflight passed"
                    if not issues
                    else "Offline pipeline preflight found missing requirements"
                ),
                "live_authority_granted": False,
            },
        }
