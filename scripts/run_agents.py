from __future__ import annotations

from pathlib import Path
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_ops.architect_agent import ArchitectAgent
from src.agent_ops.coordinator import Coordinator
from src.agent_ops.evaluator_agent import EvaluatorAgent
from src.agent_ops.trainer_agent import TrainerAgent


def main() -> None:
    agents = {
        "architect": ArchitectAgent(),
        "trainer": TrainerAgent(project_root=PROJECT_ROOT),
        "evaluator": EvaluatorAgent(),
    }
    coordinator = Coordinator(agents)

    payload = {
        "train_config": {
            "mode": "script",
            "script_path": "scripts/train.py",
            "args": [],
        },
        "model_dir": "artifacts",
    }

    result = coordinator.run("train_pipeline", payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
