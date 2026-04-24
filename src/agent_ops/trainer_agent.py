from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time
from typing import Any


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
    """

    def __init__(
        self,
        name: str = "trainer",
        project_root: str | Path = ".",
        logs_dir: str | Path = "logs/agent_ops",
        default_python: str = "python",
    ) -> None:
        self.name = name
        self.project_root = Path(project_root).resolve()
        self.logs_dir = (self.project_root / logs_dir).resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.default_python = default_python

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
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

    def _resolve_command(self, data: dict[str, Any]) -> list[str]:
        if "command" in data and isinstance(data["command"], list) and data["command"]:
            return [str(item) for item in data["command"]]

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
