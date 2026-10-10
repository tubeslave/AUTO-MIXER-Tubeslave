"""Adapter for SonyResearch/diffvox.

The upstream project is MIT licensed and remains an external dependency.
This adapter never silently installs or trains it at runtime. Configure a
checkout path on a GPU/CPU worker, then invoke it explicitly in offline mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess


@dataclass(frozen=True)
class DiffVoxConfig:
    repo_path: Path
    python_executable: str = "python"
    timeout_sec: int = 1800


class DiffVoxUnavailable(RuntimeError):
    pass


class DiffVoxAdapter:
    def __init__(self, config: DiffVoxConfig):
        self.config = config

    def status(self) -> dict[str, object]:
        repo = self.config.repo_path
        required = [repo / "main.py", repo / "ito.py", repo / "cfg" / "config.yaml"]
        return {
            "available": all(p.exists() for p in required),
            "repo_path": str(repo),
            "required_files": [str(p) for p in required],
            "mode": "offline_proposal_only",
            "auto_apply": False,
        }

    def _require(self) -> None:
        if not self.status()["available"]:
            raise DiffVoxUnavailable(
                "DiffVox checkout is not available on this worker. "
                "Install SonyResearch/diffvox and its requirements first."
            )

    def retrieve_effects(self, data_dir: str, log_dir: str, extra_args: list[str] | None = None) -> dict[str, object]:
        """Run upstream parameter retrieval on paired vocal data.

        This is intentionally an offline proposal step. It does not write any
        live-console DSP and does not mark resulting parameters as accepted.
        """
        self._require()
        cmd = [
            self.config.python_executable,
            "main.py",
            f"data_dir={data_dir}",
            f"log_dir={log_dir}",
        ]
        if extra_args:
            cmd.extend(extra_args)
        proc = subprocess.run(
            cmd,
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_sec,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "accepted": False,
            "requires_ab_review": True,
        }

    def inference_time_optimisation(
        self,
        selected_runs: str,
        preset_dir: str,
        output_dir: str,
        encoder: str = "afx-rep",
        weight: float = 0.1,
    ) -> dict[str, object]:
        """Run upstream Gaussian-prior ITO for vocal effect style transfer."""
        self._require()
        cmd = [
            self.config.python_executable,
            "-W", "ignore",
            "ito.py",
            selected_runs,
            preset_dir,
            output_dir,
            "--config", "presets/fx_config.yaml",
            "--method", "ito",
            "--encoder", encoder,
            "--weight", str(weight),
        ]
        proc = subprocess.run(
            cmd,
            cwd=self.config.repo_path,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_sec,
            check=False,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout[-12000:],
            "stderr": proc.stderr[-12000:],
            "accepted": False,
            "requires_ab_review": True,
        }
