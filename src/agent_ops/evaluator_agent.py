from __future__ import annotations

from pathlib import Path
from typing import Any


class EvaluatorAgent:
    def __init__(self, name: str = "evaluator") -> None:
        self.name = name

    def run(self, data: dict[str, Any]) -> dict[str, Any]:
        model_dir = data.get("model_dir")
        exists = bool(model_dir and Path(model_dir).exists())
        return {
            "agent": self.name,
            "status": "ok",
            "result": {
                "model_dir": model_dir,
                "model_dir_exists": exists,
                "summary": "Basic evaluation placeholder",
                "next_recommendation": "Attach metrics parser and listening-test pipeline.",
            },
        }
