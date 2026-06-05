from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEMORY_TOOLS = ROOT / "tools" / "memory"


def load_memory_lint():
    spec = importlib.util.spec_from_file_location(
        "memory_lint", MEMORY_TOOLS / "memory_lint.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def copy_template(tmp_path: Path, name: str) -> None:
    source = ROOT / ".ai" / "pipeline" / "templates" / name
    target = tmp_path / ".ai" / "pipeline" / "templates" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def run_tool(script: str, *args: str, root: Path | None = None) -> subprocess.CompletedProcess:
    command = [sys.executable, str(MEMORY_TOOLS / script)]
    if root is not None:
        command.extend(["--root", str(root)])
    command.extend(args)
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)


def test_required_memory_files_exist() -> None:
    memory_lint = load_memory_lint()

    missing = [
        item.path
        for item in memory_lint.REQUIRED_FILES
        if not (ROOT / item.path).is_file()
    ]

    assert missing == []


def test_memory_lint_runs_without_errors() -> None:
    result = run_tool("memory_lint.py", "--root", str(ROOT))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Status: PASS" in result.stdout


def test_memory_files_have_required_structure() -> None:
    memory_lint = load_memory_lint()
    ok, errors = memory_lint.run_lint(ROOT)

    assert ok, errors


def test_current_tasks_memory_preserves_live_readiness_boundary() -> None:
    memory_lint = load_memory_lint()
    current_tasks = next(
        item
        for item in memory_lint.REQUIRED_FILES
        if item.path == ".ai/project-layer/current_tasks.md"
    )

    assert "## Active Tasks" in current_tasks.headings
    assert "## Watch List" in current_tasks.headings
    assert "dry-run or audit-only" in current_tasks.required_terms
    assert (
        "do not treat shadow-only or offline-only evidence as live readiness"
        in current_tasks.required_terms
    )


def test_append_decision_preserves_existing_content(tmp_path: Path) -> None:
    copy_template(tmp_path, "decision_record.md")
    decisions = tmp_path / ".ai" / "project-layer" / "decisions.md"
    decisions.parent.mkdir(parents=True, exist_ok=True)
    decisions.write_text("# Decisions\n\n## Records\n\nOLD_DECISION\n", encoding="utf-8")

    result = run_tool(
        "append_decision.py",
        "--title",
        "Test decision",
        "--decision",
        "Append only.",
        "--context",
        "Unit test.",
        "--tests",
        "pytest",
        root=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = decisions.read_text(encoding="utf-8")
    assert "OLD_DECISION" in text
    assert "Test decision" in text
    assert "Append only." in text


def test_append_session_report_preserves_existing_content(tmp_path: Path) -> None:
    copy_template(tmp_path, "session_report.md")
    reports = tmp_path / ".ai" / "project-layer" / "session_reports.md"
    reports.parent.mkdir(parents=True, exist_ok=True)
    reports.write_text("# Session Reports\n\n## Records\n\nOLD_REPORT\n", encoding="utf-8")

    result = run_tool(
        "append_session_report.py",
        "--title",
        "Test report",
        "--summary",
        "Session summary.",
        "--tests",
        "pytest",
        "--memory-lint",
        "pass",
        root=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = reports.read_text(encoding="utf-8")
    assert "OLD_REPORT" in text
    assert "Test report" in text
    assert "Session summary." in text


def test_capture_note_preserves_existing_experiments(tmp_path: Path) -> None:
    experiments = tmp_path / ".ai" / "project-layer" / "experiments.md"
    experiments.parent.mkdir(parents=True, exist_ok=True)
    experiments.write_text("# Experiments\n\n## Records\n\nOLD_EXPERIMENT\n", encoding="utf-8")

    result = run_tool(
        "capture_note.py",
        "--title",
        "Captured note",
        "--note",
        "Raw observation.",
        "--mode",
        "dry-run",
        root=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    text = experiments.read_text(encoding="utf-8")
    assert "OLD_EXPERIMENT" in text
    assert "Captured note" in text
    assert "Raw observation." in text


def test_safety_files_forbid_live_osc_writes_without_approval() -> None:
    paths = [
        ROOT / ".ai" / "agent-core" / "safety.md",
        ROOT / ".ai" / "project-layer" / "wing_osc_policy.md",
        ROOT / ".ai" / "project-layer" / "live_safety_rules.md",
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "live osc writes are forbidden without explicit approval" in text
        assert "live wing writes are forbidden without explicit approval" in text


def test_audit_dry_run_and_live_write_modes_are_separated() -> None:
    text = "\n".join(
        [
            (ROOT / ".ai" / "agent-core" / "safety.md").read_text(encoding="utf-8"),
            (ROOT / ".ai" / "project-layer" / "wing_osc_policy.md").read_text(
                encoding="utf-8"
            ),
            (ROOT / ".ai" / "project-layer" / "live_safety_rules.md").read_text(
                encoding="utf-8"
            ),
        ]
    ).lower()

    for mode in ("audit-only", "dry-run", "supervised-write", "live-write"):
        assert mode in text

    assert text.index("audit-only") != text.index("dry-run")
    assert text.index("dry-run") != text.index("live-write")
