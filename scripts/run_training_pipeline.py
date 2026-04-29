from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_ops.evaluator_agent import EvaluatorAgent
from src.agent_ops.trainer_agent import TrainerAgent


def main() -> None:
    trainer = TrainerAgent(project_root=PROJECT_ROOT)
    evaluator = EvaluatorAgent()

    training_result = trainer.run(
        {"experiments_config": "configs/training/experiments.yaml"}
    )
    evaluation_result = evaluator.run({"best_run": training_result.get("best_run")})
    result = {
        "training": training_result,
        "evaluation": evaluation_result,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
