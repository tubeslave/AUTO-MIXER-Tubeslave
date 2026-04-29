from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

from src.agent_ops.evaluator_agent import EvaluatorAgent
from src.agent_ops.trainer_agent import TrainerAgent


def test_trainer_agent_runs_yaml_experiments_and_selects_lowest_loss(tmp_path: Path):
    config_dir = tmp_path / "configs" / "training"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "experiments.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "name": "high_loss",
                        "command": [
                            sys.executable,
                            "-c",
                            "print('loss=0.8000 accuracy=0.10')",
                        ],
                    },
                    {
                        "name": "low_loss",
                        "command": [
                            sys.executable,
                            "-c",
                            "print('loss=0.2000 accuracy=0.90')",
                        ],
                    },
                ],
                "selection_metric": "loss",
                "selection_mode": "min",
            }
        ),
        encoding="utf-8",
    )

    agent = TrainerAgent(project_root=tmp_path, runs_dir="runs")

    result = agent.run({"experiments_config": "configs/training/experiments.yaml"})

    assert result["status"] == "ok"
    assert result["best_run"]["name"] == "low_loss"
    assert result["best_run"]["metrics"]["loss"] == 0.2
    assert Path(result["summary_path"]).exists()
    assert Path(result["best_run"]["run_dir"], "metadata.json").exists()
    assert Path(result["best_run"]["log_path"]).exists()


def test_evaluator_agent_writes_evaluation_for_best_run(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log_path = run_dir / "train.log"
    log_path.write_text("loss=0.2\n", encoding="utf-8")
    best_run = {
        "run_id": "test_run",
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "success": True,
        "metrics": {"loss": 0.2},
    }

    result = EvaluatorAgent().run({"best_run": best_run})

    assert result["status"] == "ok"
    assert result["evaluated_run"] == "test_run"
    assert result["checks"]["run_dir_exists"] is True
    assert result["checks"]["log_exists"] is True
    evaluation_path = Path(result["evaluation_path"])
    assert evaluation_path.exists()
    assert json.loads(evaluation_path.read_text(encoding="utf-8"))["metrics"] == {
        "loss": 0.2
    }
