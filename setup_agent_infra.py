#!/usr/bin/env python3
"""Idempotent setup for a standalone multi-agent template layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parent


@dataclass
class SetupReport:
    created: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def record_created(self, path: Path) -> None:
        self.created.append(_display_path(path))

    def record_skipped(self, path: Path) -> None:
        self.skipped.append(_display_path(path))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _resolve_relative_path(relative_path: str) -> Path:
    """
    Reuse existing directory casing when the filesystem is case-insensitive.

    This keeps reports aligned with the real project layout, e.g. `Docs/`
    instead of creating a parallel lowercase entry in the output.
    """
    current = ROOT
    for part in Path(relative_path).parts:
        if current.exists():
            match = next(
                (child.name for child in current.iterdir() if child.name.lower() == part.lower()),
                None,
            )
            current = current / (match or part)
        else:
            current = current / part
    return current


def _normalize_content(content: str) -> str:
    return textwrap.dedent(content).strip() + "\n"


def ensure_dir(report: SetupReport, relative_path: str) -> None:
    path = _resolve_relative_path(relative_path)
    if path.exists():
        report.record_skipped(path)
        return
    path.mkdir(parents=True, exist_ok=True)
    report.record_created(path)


def safe_write(report: SetupReport, relative_path: str, content: str) -> None:
    path = _resolve_relative_path(relative_path)
    if path.exists():
        report.record_skipped(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_normalize_content(content), encoding="utf-8")
    report.record_created(path)


def build_templates() -> dict[str, str]:
    return {
        "docs/agent_responsibilities.md": """
            # Agent Responsibilities

            > Этот шаблонный слой создаётся отдельно от текущих runtime-агентов
            > проекта и не вмешивается в `backend/agents`.

            ## mixing_agent
            Отвечает за сведение (gain, EQ, compression, routing)

            ## architect_agent
            Проектирует систему агентов и их взаимодействие

            ## trainer_agent
            Организует обучение моделей и пайплайны

            ## evaluator_agent
            Проверяет качество (метрики + субъективная оценка)

            ## coordinator
            Оркестрирует агентов и распределяет задачи
            """,
        "prompts/architect.md": """
            Ты — AI архитектор системы агентов для концертной звукорежиссуры.

            Задача:
            - разбивать систему на специализированных агентов
            - описывать их интерфейсы (input/output)
            - предлагать улучшения архитектуры

            Правила:
            - не усложняй
            - делай модульно
            - всегда объясняй решения
            """,
        "prompts/trainer.md": """
            Ты — AI специалист по обучению моделей звукорежиссуры.

            Задача:
            - создавать pipeline обучения
            - предлагать датасеты
            - выбирать loss функции
            - отслеживать прогресс обучения

            Правила:
            - начинай с простых моделей
            - избегай избыточной сложности
            """,
        "prompts/evaluator.md": """
            Ты — AI evaluator музыкальных моделей.

            Задача:
            - оценивать качество микса
            - проверять стабильность
            - находить ошибки

            Метрики:
            - loudness
            - clipping
            - vocal clarity
            - balance
            """,
        "configs/agents/architect.yaml": """
            name: architect_agent
            model: gpt-4o-mini
            temperature: 0.2
            """,
        "configs/agents/trainer.yaml": """
            name: trainer_agent
            model: gpt-4o-mini
            temperature: 0.3
            """,
        "configs/agents/evaluator.yaml": """
            name: evaluator_agent
            model: gpt-4o-mini
            temperature: 0.1
            """,
        "src/__init__.py": """
            \"\"\"Standalone agent template package.\"\"\"
            """,
        "src/agent_ops/__init__.py": """
            \"\"\"Template agent implementations for multi-agent experiments.\"\"\"
            """,
        "src/multi_agent/__init__.py": """
            \"\"\"Namespace for future multi-agent orchestration utilities.\"\"\"
            """,
        "src/agent_ops/base_agent.py": """
            class BaseAgent:
                def __init__(self, name):
                    self.name = name

                def run(self, input_data):
                    raise NotImplementedError
            """,
        "src/agent_ops/architect_agent.py": """
            from .base_agent import BaseAgent


            class ArchitectAgent(BaseAgent):
                def run(self, project_state):
                    return "Suggest agent architecture improvements"
            """,
        "src/agent_ops/trainer_agent.py": """
            from .base_agent import BaseAgent


            class TrainerAgent(BaseAgent):
                def run(self, dataset):
                    return "Suggest training pipeline"
            """,
        "src/agent_ops/evaluator_agent.py": """
            from .base_agent import BaseAgent


            class EvaluatorAgent(BaseAgent):
                def run(self, model_output):
                    return "Evaluate mix quality"
            """,
        "src/agent_ops/coordinator.py": """
            class Coordinator:
                def __init__(self, agents):
                    self.agents = agents

                def run(self, task, data):
                    results = {}
                    for name, agent in self.agents.items():
                        results[name] = agent.run(data)
                    return results
            """,
        "scripts/run_agents.py": """
            from pathlib import Path
            import sys


            PROJECT_ROOT = Path(__file__).resolve().parents[1]
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.agent_ops.architect_agent import ArchitectAgent
            from src.agent_ops.trainer_agent import TrainerAgent
            from src.agent_ops.evaluator_agent import EvaluatorAgent
            from src.agent_ops.coordinator import Coordinator


            def main():
                agents = {
                    "architect": ArchitectAgent("architect"),
                    "trainer": TrainerAgent("trainer"),
                    "evaluator": EvaluatorAgent("evaluator"),
                }
                coordinator = Coordinator(agents)
                result = coordinator.run("analyze_project", {})
                print(result)


            if __name__ == "__main__":
                main()
            """,
    }


def print_report(report: SetupReport) -> None:
    print(f"Project root: {ROOT}")

    print("\n=== CREATED ===")
    if report.created:
        for item in sorted(report.created):
            print(" +", item)
    else:
        print(" (none)")

    print("\n=== SKIPPED (already exists) ===")
    if report.skipped:
        for item in sorted(report.skipped):
            print(" =", item)
    else:
        print(" (none)")

    print(
        f"\nDone. Created: {len(report.created)} | "
        f"Skipped: {len(report.skipped)}"
    )


def main() -> None:
    report = SetupReport()

    dirs = [
        "src",
        "src/agent_ops",
        "src/multi_agent",
        "configs",
        "configs/agents",
        "prompts",
        "docs",
    ]
    for directory in dirs:
        ensure_dir(report, directory)

    for relative_path, content in build_templates().items():
        safe_write(report, relative_path, content)

    print_report(report)


if __name__ == "__main__":
    main()
