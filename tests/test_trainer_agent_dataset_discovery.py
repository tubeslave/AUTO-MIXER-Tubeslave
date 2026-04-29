from pathlib import Path

from src.agent_ops.trainer_agent import TrainerAgent


def test_trainer_agent_builds_dataset_discovery_command(tmp_path: Path):
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
