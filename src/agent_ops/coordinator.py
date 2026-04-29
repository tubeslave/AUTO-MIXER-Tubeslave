from __future__ import annotations

from typing import Any


class Coordinator:
    def __init__(self, agents: dict[str, Any]) -> None:
        self.agents = agents

    def run(self, task: str, data: dict[str, Any]) -> dict[str, Any]:
        results = {}
        for name, agent in self.agents.items():
            try:
                results[name] = agent.run(data)
            except Exception as exc:
                results[name] = {
                    "agent": name,
                    "status": "failed",
                    "error": str(exc),
                }
        return {
            "task": task,
            "results": results,
        }
