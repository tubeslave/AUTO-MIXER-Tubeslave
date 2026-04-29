from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import yaml


@dataclass
class TrainingResult:
    success: bool
    returncode: int
    command: list[str]
    started_at: float
    finished_at: float
    duration_sec: float
    stdout_tail: str
    stderr_tail: str
    log_path: str | None = None


class TrainerAgent:
    """
    Wrapper around the project's training pipeline.

    Accepts either:
    - an explicit command in data["command"]
    - or a train_config describing script/module execution
    - or task="discover_datasets" to bootstrap sound-engineering datasets
    - or experiments_config pointing to a YAML experiment suite
    """

    def __init__(
        self,
        name: str = "trainer",
        project_root: str | Path = ".",
        logs_dir: str | Path = "logs/agent_ops",
        runs_dir: str | Path = "artifacts/runs",
        default_python: str = "python",
    ) -> None:
        self.name = name
        self.project_root = Path(project_root).resolve()
        self.logs_dir = (self.project_root / logs_dir).resolve()
        self.runs_dir = (self.project_root / runs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.default_python = default_python

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        if "experiments_config" in data:
            return self._run_experiment_suite(data)

        command = self._resolve_command(data)
        started_at = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = self.logs_dir / f"trainer_run_{timestamp}.log"

        process = subprocess.run(
            command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            shell=False,
        )

        finished_at = time.time()
        stdout = process.stdout or ""
        stderr = process.stderr or ""

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND:\n")
            log_file.write(" ".join(command) + "\n\n")
            log_file.write("STDOUT:\n")
            log_file.write(stdout + "\n\n")
            log_file.write("STDERR:\n")
            log_file.write(stderr + "\n")

        result = TrainingResult(
            success=process.returncode == 0,
            returncode=process.returncode,
            command=command,
            started_at=started_at,
            finished_at=finished_at,
            duration_sec=finished_at - started_at,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            log_path=str(log_path),
        )

        return {
            "agent": self.name,
            "status": "ok" if result.success else "failed",
            "result": self._to_dict(result),
        }

    def _run_experiment_suite(self, data: dict[str, Any]) -> dict[str, Any]:
        config_path = data.get("experiments_config", "configs/training/experiments.yaml")
        config = self._load_yaml(config_path)
        experiments = config.get("experiments", [])
        if not isinstance(experiments, list):
            raise ValueError("experiments must be a list")

        selection_metric = str(config.get("selection_metric", "loss"))
        selection_mode = str(config.get("selection_mode", "min")).lower()

        results = [self._run_experiment(exp) for exp in experiments]
        best_run = self._select_best(results, selection_metric, selection_mode)
        summary = {
            "agent": self.name,
            "status": "ok",
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_run": best_run,
            "runs": results,
        }
        summary_path = self.runs_dir / "training_summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["summary_path"] = str(summary_path)
        return summary

    def _run_experiment(self, exp: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(exp, dict):
            raise ValueError("Each experiment must be a dict")

        name = str(exp["name"])
        command = [str(item) for item in exp["command"]]
        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{name}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        log_path = run_dir / "train.log"
        metadata_path = run_dir / "metadata.json"
        started_at = time.time()
        process = subprocess.run(
            command,
            cwd=str(self.project_root),
            capture_output=True,
            text=True,
            shell=False,
        )
        finished_at = time.time()
        stdout = process.stdout or ""
        stderr = process.stderr or ""

        log_path.write_text(
            "COMMAND:\n"
            + " ".join(command)
            + "\n\nSTDOUT:\n"
            + stdout
            + "\n\nSTDERR:\n"
            + stderr,
            encoding="utf-8",
        )

        metrics = self._parse_metrics(stdout + "\n" + stderr)
        result = {
            "name": name,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "command": command,
            "success": process.returncode == 0,
            "returncode": process.returncode,
            "duration_sec": round(finished_at - started_at, 3),
            "metrics": metrics,
            "log_path": str(log_path),
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        }

        metadata_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return result

    def _resolve_command(self, data: dict[str, Any]) -> list[str]:
        if "command" in data and isinstance(data["command"], list) and data["command"]:
            return [str(item) for item in data["command"]]

        task = str(data.get("task", "")).strip().lower()
        if task in {"discover_datasets", "dataset_discovery"}:
            return self._resolve_dataset_discovery_command(data)

        cfg = data.get("train_config", {})
        if not isinstance(cfg, dict):
            raise ValueError("train_config must be a dict")

        mode = cfg.get("mode", "script")
        if mode == "script":
            script_path = cfg.get("script_path", "scripts/train.py")
            extra_args = cfg.get("args", [])
            return [self.default_python, script_path, *map(str, extra_args)]

        if mode == "module":
            module_name = cfg.get("module_name")
            if not module_name:
                raise ValueError("train_config.module_name is required when mode='module'")
            extra_args = cfg.get("args", [])
            return [self.default_python, "-m", module_name, *map(str, extra_args)]

        raise ValueError(f"Unsupported train mode: {mode}")

    def _resolve_dataset_discovery_command(self, data: dict[str, Any]) -> list[str]:
        cfg = data.get("dataset_discovery", {})
        if not isinstance(cfg, dict):
            raise ValueError("dataset_discovery must be a dict")

        script_path = cfg.get("script_path", "scripts/discover_training_datasets.py")
        command = [self.default_python, str(script_path)]

        output_dir = cfg.get("output_dir")
        if output_dir:
            command.extend(["--output-dir", str(output_dir)])

        report_path = cfg.get("report_path")
        if report_path:
            command.extend(["--report", str(report_path)])

        max_dataset_bytes = cfg.get("max_dataset_bytes")
        if max_dataset_bytes is not None:
            command.extend(["--max-dataset-bytes", str(max_dataset_bytes)])

        timeout_sec = cfg.get("timeout_sec")
        if timeout_sec is not None:
            command.extend(["--timeout-sec", str(timeout_sec)])

        search_limit = cfg.get("search_limit")
        if search_limit is not None:
            command.extend(["--search-limit", str(search_limit)])

        for dataset_id in cfg.get("dataset_ids", []) or []:
            command.extend(["--dataset-id", str(dataset_id)])

        for query in cfg.get("search_queries", []) or []:
            command.extend(["--search-query", str(query)])

        if cfg.get("download") is False:
            command.append("--no-download")

        if cfg.get("convert_jsonl") is False:
            command.append("--no-convert-jsonl")

        return command

    def _parse_metrics(self, text: str) -> dict[str, float]:
        """
        Parse simple metric tokens emitted as key=value pairs.

        Examples include loss=0.1234, val_loss=0.2345, and accuracy=0.91.
        """
        metrics: dict[str, float] = {}
        for token in text.replace(",", " ").split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
        return metrics

    def _select_best(
        self,
        results: list[dict[str, Any]],
        metric: str,
        mode: str,
    ) -> dict[str, Any] | None:
        if mode not in {"min", "max"}:
            raise ValueError("selection_mode must be 'min' or 'max'")

        valid = [
            result
            for result in results
            if result.get("success") and metric in result.get("metrics", {})
        ]
        if not valid:
            return None

        return sorted(
            valid,
            key=lambda result: result["metrics"][metric],
            reverse=mode == "max",
        )[0]

    def _load_yaml(self, path: str | Path) -> dict[str, Any]:
        full_path = self.project_root / path
        if not full_path.exists():
            raise FileNotFoundError(f"Config not found: {full_path}")

        with full_path.open("r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file) or {}

        if not isinstance(config, dict):
            raise ValueError("Experiment config must be a YAML mapping")
        return config

    @staticmethod
    def _to_dict(result: TrainingResult) -> dict[str, Any]:
        return {
            "success": result.success,
            "returncode": result.returncode,
            "command": result.command,
            "started_at": result.started_at,
            "finished_at": result.finished_at,
            "duration_sec": round(result.duration_sec, 3),
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
            "log_path": result.log_path,
        }
