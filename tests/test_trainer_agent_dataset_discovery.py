from pathlib import Path
import subprocess

from src.agent_ops import trainer_agent
from src.agent_ops.trainer_agent import TrainerAgent


def test_trainer_agent_builds_dataset_discovery_command(tmp_path: Path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "discover_training_datasets.py").write_text("print('discovery')\n")
    agent = TrainerAgent(project_root=tmp_path, default_python="python3")

    command = agent._resolve_command(
        {
            "task": "discover_datasets",
            "dataset_discovery": {
                "report_path": "/tmp/report.md",
                "output_dir": "models/training_datasets/audio_mixing",
                "dataset_ids": ["mclemcrew/MixAssist"],
                "search_queries": ["music mixing parameters"],
                "max_dataset_bytes": 1234,
                "timeout_sec": 5,
                "search_limit": 2,
                "download": False,
                "convert_jsonl": False,
            },
        }
    )

    assert command[:2] == ["python3", "scripts/discover_training_datasets.py"]
    assert "--report" in command
    assert "/tmp/report.md" in command
    assert "--dataset-id" in command
    assert "mclemcrew/MixAssist" in command
    assert "--search-query" in command
    assert "music mixing parameters" in command
    assert "--max-dataset-bytes" in command
    assert "1234" in command
    assert "--no-download" in command
    assert "--no-convert-jsonl" in command


def test_trainer_agent_records_command_timeout(tmp_path: Path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train.py").write_text("print('training')\n")
    agent = TrainerAgent(project_root=tmp_path, default_timeout_sec=2.5)
    seen = {}

    def fake_run(command, cwd, capture_output, text, shell, timeout):
        seen["timeout"] = timeout
        raise subprocess.TimeoutExpired(command, timeout, output="partial", stderr=b"slow")

    monkeypatch.setattr(trainer_agent.subprocess, "run", fake_run)

    result = agent.run({"train_config": {"script_path": "scripts/train.py"}})

    assert seen["timeout"] == 2.5
    assert result["status"] == "failed"
    assert result["result"]["returncode"] == trainer_agent.TRAINING_TIMEOUT_RETURNCODE
    assert "partial" in result["result"]["stdout_tail"]
    assert "slow" in result["result"]["stderr_tail"]
    assert "timed out after 2.5s" in result["result"]["stderr_tail"]


def test_trainer_experiment_uses_per_experiment_timeout(tmp_path: Path, monkeypatch):
    agent = TrainerAgent(project_root=tmp_path, default_timeout_sec=100.0)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "train.py").write_text("print('training')\n")
    config_dir = tmp_path / "configs" / "training"
    config_dir.mkdir(parents=True)
    config = config_dir / "experiments.yaml"
    config.write_text(
        """
experiments:
  - name: short
    command: [python, scripts/train.py]
    timeout_sec: 3
selection_metric: loss
selection_mode: min
""",
        encoding="utf-8",
    )
    seen = {}

    def fake_run(command, cwd, capture_output, text, shell, timeout):
        seen["timeout"] = timeout
        return subprocess.CompletedProcess(command, 0, stdout="loss=0.1", stderr="")

    monkeypatch.setattr(trainer_agent.subprocess, "run", fake_run)

    result = agent.run({"experiments_config": "configs/training/experiments.yaml"})

    assert seen["timeout"] == 3.0
    assert result["status"] == "ok"
    assert result["best_run"]["metrics"]["loss"] == 0.1
