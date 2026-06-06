import importlib.util
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = PROJECT_ROOT / "tools" / "codex_autoloop" / "validate_cycle_report.py"


def load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_cycle_report", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_report(project_root: Path, name: str, body: str) -> Path:
    linked_file = project_root / "tools" / "codex_autoloop" / "validate_cycle_report.py"
    linked_file.parent.mkdir(parents=True, exist_ok=True)
    linked_file.write_text("# report validator fixture\n", encoding="utf-8")

    reports_dir = project_root / ".paperclip" / "reports"
    reports_dir.mkdir(parents=True)
    report_path = reports_dir / name
    report_path.write_text(
        body.replace("{PROJECT_ROOT}", str(project_root)),
        encoding="utf-8",
    )
    return report_path


VALID_REPORT = """# Automixer Codex Cycle 20260530T000000Z

## Scope

Codex CLI was the only agent workflow used.
External Paperclip/PiperClip was not used as an agent runner.
No neighboring repositories or external local projects were analyzed, modified,
or used.
Repo-local `.paperclip/reports/` was used only as an Automixer report directory.
Repo-local report path:
`{PROJECT_ROOT}/.paperclip/reports`.

Live safety status: no live OSC/WING writes were executed.
No real console was contacted.
No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,
`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.
No `set_trim`, `set_input_trim`, `set_channel_trim`, `set_preamp_trim`, or
`set_preamp_gain` command was run.
No `set_insert`, `set_fx_model`, `set_fx_on`, `set_fx_mix`, or
`set_fx_parameter` command was run.
No commit, push, reset, destructive cleanup, secret access, or user-file deletion
was performed.
No `.env`, tokens, credentials, API keys, or local secrets were accessed or modified.

## What Was Found

Branch: `codex/project-structure-integration-20260518`.
`git status --short` showed a dirty worktree with modified files and untracked
files preserved.
Latest repo-local `.paperclip/reports/` and `Docs/reports/` entries were
inspected.
Latest test artifacts under `artifacts/` and `sessions/` were inspected.
TODO/FIXME scan, excluding vendored/generated/reference noise, found only
bounded owned-code items.
Local project tasks in `.ai/project-layer/current_tasks.md` were inspected.
Project canon from `README.md`, `AGENTS.md`, `CLAUDE.md`,
`.ai/memory/project.md`, and `Docs/PROJECT_IMPLEMENTATION_PLAN.md` confirms
the main goal: supervised gate for live changes, with all other actions kept
read-only, proposal-only, shadow-only, or offline-only.

## Short Plan

1. Keep the change inside repo-local report tooling.
2. Run focused validator tests.

## What Was Done

A safe local report validator was checked.

## Changed Files

- [tools/codex_autoloop/validate_cycle_report.py](../../tools/codex_autoloop/validate_cycle_report.py)
- [automixer_codex_cycle_20260530T000000Z.md](automixer_codex_cycle_20260530T000000Z.md)

## Tests And Checks

- `python3 -m pytest tests/test_codex_cycle_report_validator.py -q`
  - Passed: fixture validator tests.
- `python3 tools/memory/memory_lint.py`
  - Passed: required memory files and safety wording are present.

## Memory Lint Result

`python3 tools/memory/memory_lint.py` passed.
Status: `PASS`.

## What Remains

Hardware proof remains separate.

## Risks

Offline evidence is not live readiness.

## Три обязательных вопроса

1. Что нового проект научился делать?

Проект научился проверять обязательную форму безопасного отчета.

2. Что сейчас является самым слабым местом?

Самое слабое место - разрыв между offline evidence и live hardware proof.

3. Что даст наибольший прогресс в следующем цикле?

Наибольший прогресс даст один сфокусированный безопасный test/report slice.

## Объяснение простым языком

### Что было сделано

Проверена форма отчета.

### Зачем это нужно проекту

Чтобы безопасные циклы оставляли одинаковые доказательства.

### В каком состоянии проект сейчас

Уже работает безопасная форма отчета. Частично работает только локальная
проверка отчета. Еще не реализовано live hardware proof. Главные риски остаются
в разрыве между offline evidence и реальным пультом.

### Какие проблемы были обнаружены

Проблем не обнаружено.

### Что нужно сделать дальше

1. Продолжать безопасные проверки.

### Общая готовность направления

- Исследование: 88%
- Архитектура: 84%
- Реализация: 78%
- Тестирование: 91%
- Готовность к реальному концерту: 24%
"""


def test_cycle_report_validator_accepts_required_repo_local_report(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT,
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []
    assert validator.main([str(report_path)], project_root=tmp_path) == 0


def test_cycle_report_validator_requires_exact_git_status_short_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("`git status --short` showed", "`git status` showed"),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "What Was Found must include exact worktree command evidence: "
        "git status --short"
        in errors
    )


def test_cycle_report_validator_requires_current_branch_value(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Branch: `codex/project-structure-integration-20260518`.",
            "Branch:",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "What Was Found must include current branch evidence, "
        "e.g. Branch: `branch-name`"
        in errors
    )


def test_cycle_report_validator_rejects_external_report_path(tmp_path):
    validator = load_validator_module()
    external_dir = tmp_path / "paperclip" / "reports"
    external_dir.mkdir(parents=True)
    external_report = external_dir / "automixer_codex_cycle_20260530T000000Z.md"
    external_report.write_text(VALID_REPORT, encoding="utf-8")

    errors = validator.validate_report(external_report, project_root=tmp_path)

    assert any("report must be inside" in error for error in errors)


def test_cycle_report_validator_rejects_missing_safety_and_explanation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        "# Automixer Codex Cycle 20260530T000000Z\n\n## What Was Done\n\nIncomplete.\n",
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any("no live OSC/WING writes were executed" in error for error in errors)
    assert any("## Объяснение простым языком" in error for error in errors)
    assert any("1. Что нового проект научился делать?" in error for error in errors)


def test_cycle_report_validator_rejects_missing_scope_section(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## Scope\n\n"
            "Codex CLI was the only agent workflow used.\n"
            "External Paperclip/PiperClip was not used as an agent runner.\n"
            "No neighboring repositories or external local projects were analyzed, modified,\n"
            "or used.\n"
            "Repo-local `.paperclip/reports/` was used only as an Automixer report directory.\n"
            "Repo-local report path:\n"
            "`{PROJECT_ROOT}/.paperclip/reports`.\n\n",
            "Codex CLI was the only agent workflow used.\n"
            "External Paperclip/PiperClip was not used as an agent runner.\n"
            "No neighboring repositories or external local projects were analyzed, modified,\n"
            "or used.\n"
            "Repo-local `.paperclip/reports/` was used only as an Automixer report directory.\n"
            "Repo-local report path:\n"
            "`{PROJECT_ROOT}/.paperclip/reports`.\n\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing required report phrase: ## Scope" in errors


def test_cycle_report_validator_rejects_missing_project_boundary_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No neighboring repositories or external local projects were analyzed, modified,\n"
            "or used.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required report phrase: No neighboring repositories or external "
        "local projects were analyzed, modified, or used."
        in errors
    )


def test_cycle_report_validator_rejects_missing_repo_local_report_path_evidence(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Repo-local report path:\n"
            "`{PROJECT_ROOT}/.paperclip/reports`.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing required report phrase: Repo-local report path:" in errors
    assert "missing required Scope evidence: Repo-local report path:" in errors
    assert any(
        "Scope must include repo-local report path under Automixer project:" in error
        for error in errors
    )


def test_cycle_report_validator_rejects_unlabeled_repo_local_report_path(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Repo-local report path:\n"
            "`{PROJECT_ROOT}/.paperclip/reports`.\n",
            "Repo-local report path:\n"
            "Repo-local report directory was reviewed separately.\n"
            "Observed path: `{PROJECT_ROOT}/.paperclip/reports`.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Scope must attach repo-local report path to label: "
        f"Repo-local report path: {tmp_path / '.paperclip' / 'reports'}"
        in errors
    )


def test_cycle_report_validator_rejects_missing_codex_cli_agent_workflow_evidence(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("Codex CLI was the only agent workflow used.\n", ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required report phrase: Codex CLI was the only agent workflow used."
        in errors
    )
    assert (
        "missing required Scope evidence: Codex CLI was the only agent workflow used."
        in errors
    )


def test_cycle_report_validator_rejects_non_exact_codex_cli_agent_workflow_line(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Codex CLI was the only agent workflow used.\n",
            "Codex CLI was the only agent workflow used. Extra scope note.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Scope must contain exact agent workflow line: "
        "Codex CLI was the only agent workflow used."
        in errors
    )
    assert (
        "missing required Scope evidence: Codex CLI was the only agent workflow used."
        not in errors
    )


def test_cycle_report_validator_rejects_codex_cli_agent_workflow_outside_scope(
    tmp_path,
):
    validator = load_validator_module()
    moved_workflow_report = VALID_REPORT.replace(
        "Codex CLI was the only agent workflow used.\n",
        "",
    ).replace(
        "A safe local report validator was checked.",
        "Codex CLI was the only agent workflow used.\n"
        "A safe local report validator was checked.",
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        moved_workflow_report,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required Scope evidence: Codex CLI was the only agent "
        "workflow used."
        in errors
    )
    assert (
        "Scope must contain exact agent workflow line: "
        "Codex CLI was the only agent workflow used."
        in errors
    )


def test_cycle_report_validator_rejects_non_exact_external_agent_runner_line(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "External Paperclip/PiperClip was not used as an agent runner.\n",
            "External Paperclip/PiperClip was not used as an agent runner. Extra scope note.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Scope must contain exact external agent boundary line: "
        "External Paperclip/PiperClip was not used as an agent runner."
        in errors
    )
    assert (
        "missing required Scope evidence: External Paperclip/PiperClip was not "
        "used as an agent runner."
        not in errors
    )


def test_cycle_report_validator_rejects_external_agent_boundary_outside_scope(
    tmp_path,
):
    validator = load_validator_module()
    moved_boundary_report = VALID_REPORT.replace(
        "External Paperclip/PiperClip was not used as an agent runner.\n",
        "",
    ).replace(
        "A safe local report validator was checked.",
        "External Paperclip/PiperClip was not used as an agent runner.\n"
        "A safe local report validator was checked.",
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        moved_boundary_report,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required Scope evidence: External Paperclip/PiperClip was not "
        "used as an agent runner."
        in errors
    )
    assert (
        "Scope must contain exact external agent boundary line: "
        "External Paperclip/PiperClip was not used as an agent runner."
        in errors
    )


def test_cycle_report_validator_rejects_missing_secret_boundary_evidence(tmp_path):
    validator = load_validator_module()
    secret_boundary = (
        "No `.env`, tokens, credentials, API keys, or local secrets were "
        "accessed or modified.\n"
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(secret_boundary, ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required Scope evidence: No `.env`, tokens, credentials, "
        "API keys, or local secrets were accessed or modified."
        in errors
    )


def test_cycle_report_validator_rejects_missing_project_goal_evidence(tmp_path):
    validator = load_validator_module()
    goal_evidence = (
        "Project canon from `README.md`, `AGENTS.md`, `CLAUDE.md`,\n"
        "`.ai/memory/project.md`, and `Docs/PROJECT_IMPLEMENTATION_PLAN.md` confirms\n"
        "the main goal: supervised gate for live changes, with all other actions kept\n"
        "read-only, proposal-only, shadow-only, or offline-only.\n"
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(goal_evidence, ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "What Was Found is missing project-goal evidence: README.md" in errors
    assert (
        "What Was Found is missing project-goal evidence: supervised gate"
        in errors
    )
    assert (
        "What Was Found is missing project-goal evidence: "
        "read-only/proposal-only/shadow-only/offline-only boundary"
        in errors
    )


def test_cycle_report_validator_rejects_scope_secret_boundary_contradiction(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `.env`, tokens, credentials, API keys, or local secrets were accessed or modified.\n",
            "No `.env`, tokens, credentials, API keys, or local secrets were accessed or modified.\n"
            "The `.env` file was opened during setup.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Scope contradicts secret boundary evidence: .env"
        in errors
    )


def test_cycle_report_validator_rejects_body_secret_boundary_contradiction(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "Credentials and API keys were inspected during validation.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith("report body contradicts secret boundary evidence: ")
        and "credentials" in error
        and "api keys" in error
        for error in errors
    )


def test_cycle_report_validator_accepts_body_negated_secret_boundary_note(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "The `.env` file was not accessed during validation.",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_project_boundary_evidence_outside_scope(
    tmp_path,
):
    validator = load_validator_module()
    moved_boundary_report = VALID_REPORT.replace(
        "No neighboring repositories or external local projects were analyzed, modified,\n"
        "or used.\n",
        "",
    ).replace(
        "A safe local report validator was checked.",
        "No neighboring repositories or external local projects were analyzed, modified,\n"
        "or used.\n"
        "A safe local report validator was checked.",
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        moved_boundary_report,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required Scope evidence: No neighboring repositories or "
        "external local projects were analyzed, modified, or used."
        in errors
    )


def test_cycle_report_validator_rejects_external_project_path_contradiction(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "I analyzed /Users/dmitrijvolkov/paperclip to compare dispatcher state.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        "report body contradicts repo-local project boundary evidence" in error
        and "/users/dmitrijvolkov/paperclip" in error
        for error in errors
    )


def test_cycle_report_validator_rejects_external_project_read_contradiction(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "I read /Users/dmitrijvolkov/paperclip logs before choosing this task.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        "report body contradicts repo-local project boundary evidence" in error
        and "/users/dmitrijvolkov/paperclip" in error
        for error in errors
    )


def test_cycle_report_validator_rejects_external_paperclip_runner_contradiction(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "External Paperclip/PiperClip was used as an agent runner for this cycle.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        "report body contradicts repo-local project boundary evidence" in error
        and "external paperclip/piperclip" in error
        for error in errors
    )


def test_cycle_report_validator_rejects_piperclip_runner_without_external_prefix(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "PiperClip was launched as the agent runner for this cycle.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        "report body contradicts repo-local project boundary evidence" in error
        and "piperclip" in error
        for error in errors
    )


def test_cycle_report_validator_rejects_external_paperclip_scope_contradiction(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "External Paperclip/PiperClip was not used as an agent runner.",
            "External Paperclip/PiperClip was not used as an agent runner.\n"
            "External Paperclip/PiperClip was used as an agent runner after setup.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        "Scope contradicts repo-local project boundary evidence" in error
        and "external paperclip/piperclip" in error
        for error in errors
    )


def test_cycle_report_validator_rejects_live_safety_evidence_outside_scope(
    tmp_path,
):
    validator = load_validator_module()
    moved_safety_report = VALID_REPORT.replace(
        "Live safety status: no live OSC/WING writes were executed.\n",
        "",
    ).replace(
        "Branch: `codex/project-structure-integration-20260518`.",
        "Branch: `codex/project-structure-integration-20260518`.\n"
        "Live safety status: no live OSC/WING writes were executed.",
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        moved_safety_report,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing required Scope evidence: Live safety status:" in errors
    assert (
        "missing required Scope evidence: no live OSC/WING writes were executed"
        in errors
    )


def test_cycle_report_validator_rejects_missing_real_console_contact_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("No real console was contacted.\n", ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing required live-safety evidence: No real console was contacted" in errors


def test_cycle_report_validator_accepts_wrapped_real_console_contact_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No real console was contacted.",
            "No real console was\ncontacted.",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_scope_real_console_contact_contradiction(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No real console was contacted.\n",
            "No real console was contacted, but a real console was contacted "
            "during validation.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "real-console audit contradicts no-contact safety evidence: real console"
        in errors
    )


def test_cycle_report_validator_rejects_body_real_console_contact_contradiction(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "A real console was contacted during report validation.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "report body contradicts no-contact safety evidence: real console"
        in errors
    )


def test_cycle_report_validator_accepts_wrapped_required_live_write_phrase(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "no live OSC/WING writes were executed",
            "no live OSC/WING writes\nwere executed",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_missing_forbidden_command_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("`set_eq`, ", ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing forbidden command audit evidence: set_eq" in errors


def test_cycle_report_validator_rejects_missing_trim_preamp_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_trim`, `set_input_trim`, `set_channel_trim`, `set_preamp_trim`, or\n"
            "`set_preamp_gain` command was run.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith("missing forbidden command audit evidence: ")
        and "set_trim" in error
        for error in errors
    )
    assert "set_preamp_gain" in ", ".join(errors)


def test_cycle_report_validator_rejects_reset_only_in_git_reset_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "`load_snap`, `save_snap`, `mute`, or `unmute` command was run.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing forbidden command audit evidence: reset" in errors


def test_cycle_report_validator_rejects_missing_unmute_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "`load_snap`, `save_snap`, `reset`, or `mute` command was run.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing forbidden command audit evidence: unmute" in errors


def test_cycle_report_validator_rejects_missing_fx_insert_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_insert`, `set_fx_model`, `set_fx_on`, `set_fx_mix`, or\n"
            "`set_fx_parameter` command was run.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith("missing forbidden command audit evidence: ")
        and "set_insert" in error
        for error in errors
    )
    assert "set_fx_parameter" in ", ".join(errors)


def test_cycle_report_validator_rejects_unnegated_forbidden_command_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "`set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, and `unmute` were inspected as tokens.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith(
            "forbidden command audit must explicitly say command was not run: "
        )
        for error in errors
    )
    assert any("set_fader" in error for error in errors)


def test_cycle_report_validator_accepts_post_action_command_negation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "`set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, and `unmute` commands were not run.\n",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_accepts_pre_action_command_negation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "This cycle did not run `set_fader`, `set_gain`, `set_eq`,\n"
            "`set_compressor`, `route_output`, `load_snap`, `save_snap`, `reset`,\n"
            "`mute`, or `unmute` commands.\n",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_unrelated_negation_before_command_audit(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "This was not a dry run: `set_fader`, `set_gain`, `set_eq`,\n"
            "`set_compressor`, `route_output`, `load_snap`, `save_snap`, `reset`,\n"
            "`mute`, and `unmute` commands were run.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith(
            "forbidden command audit must explicitly say command was not run: "
        )
        for error in errors
    )
    assert any("set_gain" in error for error in errors)


def test_cycle_report_validator_rejects_contrasting_command_audit_clause(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "No real console was contacted, but `set_fader`, `set_gain`, `set_eq`,\n"
            "`set_compressor`, `route_output`, `load_snap`, `save_snap`, `reset`,\n"
            "`mute`, and `unmute` commands were run.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith(
            "forbidden command audit must explicitly say command was not run: "
        )
        for error in errors
    )
    assert any("set_fader" in error for error in errors)


def test_cycle_report_validator_rejects_later_affirmed_forbidden_command(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n"
            "A `set_fader` command was run during the cycle.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "forbidden command audit contradicts no-run safety evidence: set_fader"
        in errors
    )


def test_cycle_report_validator_rejects_later_affirmed_live_write(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Live safety status: no live OSC/WING writes were executed.\n",
            "Live safety status: no live OSC/WING writes were executed.\n"
            "Live OSC/WING writes were executed during the cycle.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "live write audit contradicts no-write safety evidence: "
        "live osc/wing writes"
        in errors
    )


def test_cycle_report_validator_rejects_later_sent_live_write(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Live safety status: no live OSC/WING writes were executed.\n",
            "Live safety status: no live OSC/WING writes were executed.\n"
            "Live OSC/WING writes were sent during the cycle.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "live write audit contradicts no-write safety evidence: "
        "live osc/wing writes"
        in errors
    )


def test_cycle_report_validator_rejects_later_applied_live_write(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Live safety status: no live OSC/WING writes were executed.\n",
            "Live safety status: no live OSC/WING writes were executed.\n"
            "Live OSC/WING writes were applied during the cycle.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "live write audit contradicts no-write safety evidence: "
        "live osc/wing writes"
        in errors
    )


def test_cycle_report_validator_rejects_body_affirmed_forbidden_command(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "A `set_gain` command was run during report validation.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "report body contradicts no-run safety evidence: set_gain" in errors


def test_cycle_report_validator_rejects_body_applied_forbidden_command(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "A `set_gain` command was applied during report validation.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "report body contradicts no-run safety evidence: set_gain" in errors


def test_cycle_report_validator_rejects_body_affirmed_fx_insert_command(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "A `set_fx_parameter` command was run during report validation.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "report body contradicts no-run safety evidence: set_fx_parameter"
        in errors
    )


def test_cycle_report_validator_rejects_scope_affirmed_insert_command(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n",
            "No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,\n"
            "`load_snap`, `save_snap`, `reset`, `mute`, or `unmute` command was run.\n"
            "A `set_insert` command was executed during the cycle.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "forbidden command audit contradicts no-run safety evidence: set_insert"
        in errors
    )


def test_cycle_report_validator_rejects_body_affirmed_live_write(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "Live OSC/WING writes were executed during report validation.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "report body contradicts no-write safety evidence: live osc/wing writes"
        in errors
    )


def test_cycle_report_validator_rejects_missing_non_destructive_action_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No commit, push, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any("missing non-destructive action audit evidence" in error for error in errors)
    assert any("destructive cleanup" in error for error in errors)


def test_cycle_report_validator_rejects_unnegated_non_destructive_action_audit(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No commit, push, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n",
            "No commit, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n"
            "A push was performed for release bookkeeping.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith(
            "non-destructive action audit must explicitly say action was not performed: "
        )
        for error in errors
    )
    assert any("push" in error for error in errors)


def test_cycle_report_validator_accepts_post_action_non_destructive_negation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No commit, push, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n",
            "Commit, push, reset, destructive cleanup, secret access, and user-file deletion\n"
            "were not performed.\n",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_accepts_pre_action_non_destructive_negation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No commit, push, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n",
            "This cycle did not perform commit, push, reset, destructive cleanup,\n"
            "secret access, or user-file deletion.\n",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_unrelated_negation_before_non_destructive_audit(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No commit, push, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n",
            "This was not a release: commit, push, reset, destructive cleanup,\n"
            "secret access, and user-file deletion were performed.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith(
            "non-destructive action audit must explicitly say action was not performed: "
        )
        for error in errors
    )
    assert any("push" in error for error in errors)


def test_cycle_report_validator_rejects_contrasting_non_destructive_audit_clause(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "No commit, push, reset, destructive cleanup, secret access, or user-file deletion\n"
            "was performed.\n",
            "Secret access was not performed, but commit, push, reset,\n"
            "destructive cleanup, and user-file deletion were performed.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any(
        error.startswith(
            "non-destructive action audit must explicitly say action was not performed: "
        )
        for error in errors
    )
    assert any("commit" in error for error in errors)


def test_cycle_report_validator_rejects_later_affirmed_non_destructive_action(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "was performed.\n",
            "was performed.\n"
            "A commit was performed for release bookkeeping.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "non-destructive action audit contradicts no-action evidence: commit"
        in errors
    )


def test_cycle_report_validator_rejects_body_affirmed_non_destructive_action(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "A safe local report validator was checked.",
            "A safe local report validator was checked.\n"
            "A commit was performed for report bookkeeping.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "report body contradicts no-action evidence: commit" in errors


def test_cycle_report_validator_rejects_unanswered_mandatory_questions(tmp_path):
    validator = load_validator_module()
    unanswered_questions = """1. Что нового проект научился делать?
2. Что сейчас является самым слабым местом?
3. Что даст наибольший прогресс в следующем цикле?"""
    answered_questions = """1. Что нового проект научился делать?

Проект научился проверять обязательную форму безопасного отчета.

2. Что сейчас является самым слабым местом?

Самое слабое место - разрыв между offline evidence и live hardware proof.

3. Что даст наибольший прогресс в следующем цикле?

Наибольший прогресс даст один сфокусированный безопасный test/report slice."""
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(answered_questions, unanswered_questions),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "mandatory question needs an answer: 1. Что нового проект научился делать?"
        in errors
    )
    assert (
        "mandatory question needs an answer: 2. Что сейчас является самым слабым местом?"
        in errors
    )
    assert (
        "mandatory question needs an answer: "
        "3. Что даст наибольший прогресс в следующем цикле?"
        in errors
    )


def test_cycle_report_validator_rejects_nonstandard_filename(tmp_path):
    validator = load_validator_module()
    report_path = write_report(tmp_path, "latest.md", VALID_REPORT)

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "report filename must match automixer_codex_cycle_YYYYMMDDTHHMMSSZ.md" in errors


def test_cycle_report_validator_rejects_missing_report_title(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("# Automixer Codex Cycle 20260530T000000Z\n\n", ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing report title: # Automixer Codex Cycle YYYYMMDDTHHMMSSZ" in errors


def test_cycle_report_validator_rejects_title_timestamp_mismatch(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000001Z.md",
        VALID_REPORT,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "report title timestamp must match filename: "
        "20260530T000000Z != 20260530T000001Z"
        in errors
    )


def test_cycle_report_validator_rejects_invalid_calendar_timestamp(tmp_path):
    validator = load_validator_module()
    invalid_timestamp = "20260230T250000Z"
    report_path = write_report(
        tmp_path,
        f"automixer_codex_cycle_{invalid_timestamp}.md",
        VALID_REPORT.replace("20260530T000000Z", invalid_timestamp),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "report filename timestamp must be a valid UTC timestamp: "
        f"{invalid_timestamp}"
        in errors
    )


def test_cycle_report_validator_rejects_future_timestamp_when_now_is_provided(tmp_path):
    validator = load_validator_module()
    future_timestamp = "20260531T000000Z"
    report_path = write_report(
        tmp_path,
        f"automixer_codex_cycle_{future_timestamp}.md",
        VALID_REPORT.replace("20260530T000000Z", future_timestamp),
    )

    errors = validator.validate_report(
        report_path,
        project_root=tmp_path,
        now=datetime(2026, 5, 30, 23, 59, 59, tzinfo=timezone.utc),
    )

    assert (
        "report filename timestamp must not be in the future: "
        f"{future_timestamp}"
        in errors
    )


def test_cycle_report_validator_rejects_missing_short_plan(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## Short Plan\n\n"
            "1. Keep the change inside repo-local report tooling.\n"
            "2. Run focused validator tests.\n\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "missing required short plan section: ## Short Plan or ## Plan Selected"
        in errors
    )


def test_cycle_report_validator_rejects_empty_required_evidence_section(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## What Remains\n\nHardware proof remains separate.\n\n",
            "## What Remains\n\n- \n\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "required report section has no content: ## What Remains" in errors


def test_cycle_report_validator_rejects_empty_changed_files_section(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## Changed Files\n\n"
            "- [tools/codex_autoloop/validate_cycle_report.py]"
            "(../../tools/codex_autoloop/validate_cycle_report.py)\n"
            "- [automixer_codex_cycle_20260530T000000Z.md]"
            "(automixer_codex_cycle_20260530T000000Z.md)\n\n",
            "## Changed Files\n\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "required report section has no content: ## Changed Files" in errors


def test_cycle_report_validator_rejects_empty_plain_language_subsection(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "### Зачем это нужно проекту\n\n"
            "Чтобы безопасные циклы оставляли одинаковые доказательства.\n\n",
            "### Зачем это нужно проекту\n\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "plain-language subsection has no content: ### Зачем это нужно проекту"
        in errors
    )


def test_cycle_report_validator_rejects_incomplete_project_state_explanation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Уже работает безопасная форма отчета. Частично работает только локальная\n"
            "проверка отчета. Еще не реализовано live hardware proof. Главные риски остаются\n"
            "в разрыве между offline evidence и реальным пультом.",
            "Проект остается supervised-first.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "plain-language state subsection must mention what already works" in errors
    assert "plain-language state subsection must mention what works partially" in errors
    assert "plain-language state subsection must mention what is not implemented" in errors
    assert "plain-language state subsection must mention risks" in errors


def test_cycle_report_validator_rejects_missing_discovery_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "TODO/FIXME scan, excluding vendored/generated/reference noise, found only\n"
            "bounded owned-code items.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "What Was Found is missing discovery evidence: TODO/FIXME scan" in errors


def test_cycle_report_validator_rejects_report_discovery_without_recent_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Latest repo-local `.paperclip/reports/` and `Docs/reports/` entries were\n"
            "inspected.\n",
            "Repo-local `.paperclip/reports/` and `Docs/reports/` entries were\n"
            "inspected.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "What Was Found must mention latest/recent discovery evidence: "
        "repo-local paperclip reports"
        in errors
    )
    assert "What Was Found must mention latest/recent discovery evidence: Docs reports" in errors


def test_cycle_report_validator_rejects_test_artifacts_without_recent_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Latest test artifacts under `artifacts/` and `sessions/` were inspected.",
            "Test artifacts under `artifacts/` and `sessions/` were inspected.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "What Was Found must mention latest/recent discovery evidence: test artifacts"
        in errors
    )


def test_cycle_report_validator_rejects_missing_modified_file_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`git status --short` showed a dirty worktree with modified files and untracked\n"
            "files preserved.\n",
            "`git status --short` showed a dirty worktree with untracked\n"
            "files preserved.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "What Was Found is missing discovery evidence: modified files" in errors


def test_cycle_report_validator_rejects_missing_new_file_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`git status --short` showed a dirty worktree with modified files and untracked\n"
            "files preserved.\n",
            "`git status --short` showed a dirty worktree with modified\n"
            "files preserved.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "What Was Found is missing discovery evidence: new/untracked files" in errors


def test_cycle_report_validator_rejects_todo_scan_without_boundary_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "TODO/FIXME scan, excluding vendored/generated/reference noise, found only\n"
            "bounded owned-code items.\n",
            "TODO/FIXME scan found local items.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "What Was Found TODO/FIXME scan must mention owned-code focus" in errors
    assert (
        "What Was Found TODO/FIXME scan must mention excluded "
        "vendored/generated/reference noise"
        in errors
    )


def test_cycle_report_validator_rejects_tests_without_outcome_evidence(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        (
            VALID_REPORT
            .replace("  - Passed: fixture validator tests.\n", "")
            .replace(
                "  - Passed: required memory files and safety wording are present.\n",
                "",
            )
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Tests And Checks must include outcome evidence: passed, failed, "
        "skipped, blocked, or not run"
        in errors
    )


def test_cycle_report_validator_rejects_missing_memory_lint_result(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "- `python3 tools/memory/memory_lint.py`\n"
            "  - Passed: required memory files and safety wording are present.\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "Tests And Checks must include memory lint result" in errors


def test_cycle_report_validator_rejects_memory_lint_without_outcome(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "- `python3 tools/memory/memory_lint.py`\n"
            "  - Passed: required memory files and safety wording are present.\n",
            "- `python3 tools/memory/memory_lint.py`\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Tests And Checks memory lint result must include outcome evidence: "
        "passed, failed, skipped, blocked, or not run"
        in errors
    )


def test_cycle_report_validator_rejects_tests_and_checks_memory_lint_with_alternate_root(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "- `python3 tools/memory/memory_lint.py`\n"
            "  - Passed: required memory files and safety wording are present.\n",
            "- `python3 tools/memory/memory_lint.py --root /tmp/other`\n"
            "  - Passed: required memory files and safety wording are present.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Tests And Checks memory lint result must use repo-local memory lint "
        "without alternate-root arguments: --root"
        in errors
    )


def test_cycle_report_validator_rejects_missing_memory_lint_result_section(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## Memory Lint Result\n\n"
            "`python3 tools/memory/memory_lint.py` passed.\n"
            "Status: `PASS`.\n\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing required report phrase: ## Memory Lint Result" in errors


def test_cycle_report_validator_rejects_memory_lint_result_without_outcome(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`python3 tools/memory/memory_lint.py` passed.\n"
            "Status: `PASS`.\n",
            "`python3 tools/memory/memory_lint.py` ran.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Memory Lint Result must include outcome evidence: passed, failed, "
        "skipped, blocked, or not run"
        in errors
    )


def test_cycle_report_validator_rejects_memory_lint_result_without_pass_fail_status(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`python3 tools/memory/memory_lint.py` passed.\n"
            "Status: `PASS`.\n",
            "`python3 tools/memory/memory_lint.py` passed.\n"
            "All required memory files were present.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "Memory Lint Result must include explicit PASS or FAIL status" in errors


def test_cycle_report_validator_rejects_memory_lint_result_without_exact_command(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`python3 tools/memory/memory_lint.py` passed.\n"
            "Status: `PASS`.\n",
            "`tools/memory/memory_lint.py` passed.\n"
            "Status: `PASS`.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Memory Lint Result must include exact command: "
        "python3 tools/memory/memory_lint.py"
        in errors
    )


def test_cycle_report_validator_rejects_memory_lint_result_with_alternate_root(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "`python3 tools/memory/memory_lint.py` passed.\n"
            "Status: `PASS`.\n",
            "`python3 tools/memory/memory_lint.py --root /tmp/other` passed.\n"
            "Status: `PASS`.\n",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Memory Lint Result must use repo-local memory lint without "
        "alternate-root arguments: --root"
        in errors
    )


def test_cycle_report_validator_rejects_memory_lint_result_after_what_remains(
    tmp_path,
):
    validator = load_validator_module()
    memory_lint_section = (
        "## Memory Lint Result\n\n"
        "`python3 tools/memory/memory_lint.py` passed.\n"
        "Status: `PASS`.\n\n"
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(memory_lint_section, "").replace(
            "## What Remains\n\nHardware proof remains separate.\n\n",
            "## What Remains\n\nHardware proof remains separate.\n\n"
            f"{memory_lint_section}",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "required middle sections are out of order: "
        "## Memory Lint Result must appear before ## What Remains"
        in errors
    )


def test_cycle_report_validator_accepts_plan_selected_heading(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("## Short Plan", "## Plan Selected"),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_core_sections_out_of_order(tmp_path):
    validator = load_validator_module()
    plan_section = (
        "## Short Plan\n\n"
        "1. Keep the change inside repo-local report tooling.\n"
        "2. Run focused validator tests.\n\n"
    )
    done_section = (
        "## What Was Done\n\n"
        "A safe local report validator was checked.\n\n"
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(plan_section, "").replace(
            done_section,
            f"{done_section}{plan_section}",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "required core sections are out of order: "
        "## Short Plan or ## Plan Selected must appear before ## What Was Done"
        in errors
    )


def test_cycle_report_validator_rejects_plan_without_list_items(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "1. Keep the change inside repo-local report tooling.\n"
            "2. Run focused validator tests.",
            "Keep the change inside repo-local report tooling and run focused tests.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "## Short Plan must contain between 1 and 3 list items" in errors


def test_cycle_report_validator_accepts_three_short_plan_items(tmp_path):
    validator = load_validator_module()
    plan = "\n".join([
        "1. Inspect repo-local state.",
        "2. Patch report tooling only.",
        "3. Run focused validation.",
    ])
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "1. Keep the change inside repo-local report tooling.\n"
            "2. Run focused validator tests.",
            plan,
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_more_than_three_short_plan_items(tmp_path):
    validator = load_validator_module()
    plan = "\n".join([
        "1. Inspect repo-local state.",
        "2. Patch report tooling only.",
        "3. Run focused validation.",
        "4. Start another unrelated task.",
    ])
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "1. Keep the change inside repo-local report tooling.\n"
            "2. Run focused validator tests.",
            plan,
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "## Short Plan must contain no more than 3 list items" in errors


def test_cycle_report_validator_rejects_changed_files_without_repo_local_link(tmp_path):
    validator = load_validator_module()
    body = (
        VALID_REPORT
        .replace(
            "[tools/codex_autoloop/validate_cycle_report.py](../../tools/codex_autoloop/validate_cycle_report.py)",
            "`tools/codex_autoloop/validate_cycle_report.py`",
        )
        .replace(
            "- [automixer_codex_cycle_20260530T000000Z.md](automixer_codex_cycle_20260530T000000Z.md)\n",
            "",
        )
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        body,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Changed Files section must include at least one markdown link to a repo-local file"
        in errors
    )


def test_cycle_report_validator_rejects_missing_self_report_link(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "- [automixer_codex_cycle_20260530T000000Z.md](automixer_codex_cycle_20260530T000000Z.md)\n",
            "",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "Changed Files section must include a markdown link to this cycle report"
        in errors
    )


def test_cycle_report_validator_rejects_changed_file_link_outside_project(tmp_path):
    validator = load_validator_module()
    external_target = tmp_path.parent / "outside_project.py"
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "../../tools/codex_autoloop/validate_cycle_report.py",
            str(external_target),
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any("changed file link points outside project" in error for error in errors)


def test_cycle_report_validator_rejects_nonexistent_changed_file_link(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "../../tools/codex_autoloop/validate_cycle_report.py",
            "../../tools/codex_autoloop/missing_report_validator.py",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any("changed file link does not exist" in error for error in errors)


def test_cycle_report_validator_accepts_changed_file_link_with_line_suffix(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "../../tools/codex_autoloop/validate_cycle_report.py",
            "../../tools/codex_autoloop/validate_cycle_report.py:12",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_accepts_angle_wrapped_changed_file_link_with_line_suffix(
    tmp_path,
):
    validator = load_validator_module()
    docs_file = tmp_path / "Docs" / "File With Spaces.md"
    docs_file.parent.mkdir()
    docs_file.write_text("# Fixture\n", encoding="utf-8")
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "[tools/codex_autoloop/validate_cycle_report.py](../../tools/codex_autoloop/validate_cycle_report.py)",
            "[Docs/File With Spaces.md](<../../Docs/File With Spaces.md:3>)",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_changed_file_link_outside_project_with_line_suffix(
    tmp_path,
):
    validator = load_validator_module()
    external_target = tmp_path.parent / "outside_project.py"
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "../../tools/codex_autoloop/validate_cycle_report.py",
            f"{external_target}:12",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any("changed file link points outside project" in error for error in errors)


def test_cycle_report_validator_allows_heading_reference_before_actual_section(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "Проблем не обнаружено.",
            "Проблема: validator mentioned `### Общая готовность направления` earlier.",
        ),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_missing_readiness_score(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("- Тестирование: 91%\n", ""),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing readiness score: Тестирование: N%" in errors


def test_cycle_report_validator_rejects_readiness_scores_outside_readiness_section(
    tmp_path,
):
    validator = load_validator_module()
    misplaced_scores = """## What Was Found

Branch: `codex/project-structure-integration-20260518`.
`git status --short` showed a dirty worktree with modified files and untracked
files preserved.
Latest repo-local `.paperclip/reports/` and `Docs/reports/` entries were
inspected.
Latest test artifacts under `artifacts/` and `sessions/` were inspected.
TODO/FIXME scan, excluding vendored/generated/reference noise, found only
bounded owned-code items.
Local project tasks in `.ai/project-layer/current_tasks.md` were inspected.
Project canon from `README.md`, `AGENTS.md`, `CLAUDE.md`,
`.ai/memory/project.md`, and `Docs/PROJECT_IMPLEMENTATION_PLAN.md` confirms
the main goal: supervised gate for live changes, with all other actions kept
read-only, proposal-only, shadow-only, or offline-only.
Earlier readiness notes:
- Исследование: 88%
- Архитектура: 84%
- Реализация: 78%
- Тестирование: 91%
- Готовность к реальному концерту: 24%"""
    body = (
        VALID_REPORT
        .replace(
            """## What Was Found

Branch: `codex/project-structure-integration-20260518`.
`git status --short` showed a dirty worktree with modified files and untracked
files preserved.
Latest repo-local `.paperclip/reports/` and `Docs/reports/` entries were
inspected.
Latest test artifacts under `artifacts/` and `sessions/` were inspected.
TODO/FIXME scan, excluding vendored/generated/reference noise, found only
bounded owned-code items.
Local project tasks in `.ai/project-layer/current_tasks.md` were inspected.
Project canon from `README.md`, `AGENTS.md`, `CLAUDE.md`,
`.ai/memory/project.md`, and `Docs/PROJECT_IMPLEMENTATION_PLAN.md` confirms
the main goal: supervised gate for live changes, with all other actions kept
read-only, proposal-only, shadow-only, or offline-only.""",
            misplaced_scores,
        )
        .replace(
            "- Исследование: 88%\n"
            "- Архитектура: 84%\n"
            "- Реализация: 78%\n"
            "- Тестирование: 91%\n"
            "- Готовность к реальному концерту: 24%\n",
            "",
        )
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        body,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "missing readiness score: Исследование: N%" in errors
    assert "missing readiness score: Готовность к реальному концерту: N%" in errors


def test_cycle_report_validator_rejects_out_of_range_readiness_score(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "- Готовность к реальному концерту: 24%",
            "- Готовность к реальному концерту: 101%",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "readiness score must be between 0 and 100: Готовность к реальному концерту: 101%"
        in errors
    )


def test_cycle_report_validator_accepts_five_next_step_items(tmp_path):
    validator = load_validator_module()
    next_steps = "\n".join([
        "1. Проверить lock metadata.",
        "2. Разнести dirty diff.",
        "3. Откалибровать SafeGain.",
        "4. Заменить mock telemetry.",
        "5. Повторить full test gate.",
    ])
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("1. Продолжать безопасные проверки.", next_steps),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


def test_cycle_report_validator_rejects_next_steps_without_list_items(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "1. Продолжать безопасные проверки.",
            "Продолжать безопасные проверки.",
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "### Что нужно сделать дальше must contain between 1 and 5 list items"
        in errors
    )


def test_cycle_report_validator_rejects_more_than_five_next_step_items(tmp_path):
    validator = load_validator_module()
    next_steps = "\n".join([
        "1. Проверить lock metadata.",
        "2. Разнести dirty diff.",
        "3. Откалибровать SafeGain.",
        "4. Заменить mock telemetry.",
        "5. Повторить full test gate.",
        "6. Запланировать следующий отчет.",
    ])
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("1. Продолжать безопасные проверки.", next_steps),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "### Что нужно сделать дальше must contain no more than 5 list items" in errors


def test_cycle_report_validator_rejects_out_of_order_end_sections(tmp_path):
    validator = load_validator_module()
    body = (
        VALID_REPORT
        .replace("## Три обязательных вопроса", "__QUESTIONS_SECTION__", 1)
        .replace("## Объяснение простым языком", "## Три обязательных вопроса", 1)
        .replace("__QUESTIONS_SECTION__", "## Объяснение простым языком", 1)
    )
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        body,
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert any("required end sections are out of order" in error for error in errors)


def test_cycle_report_validator_rejects_top_level_section_between_questions_and_plain_explanation(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## Объяснение простым языком",
            "## Extra Evidence\n\nExtra notes should not sit between the "
            "mandatory questions and the plain-language explanation.\n\n"
            "## Объяснение простым языком",
            1,
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "## Три обязательных вопроса must directly precede "
        "## Объяснение простым языком without another markdown heading "
        "between them"
        in errors
    )


def test_cycle_report_validator_rejects_subsection_between_questions_and_plain_explanation(
    tmp_path,
):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace(
            "## Объяснение простым языком",
            "### Extra Evidence\n\nNested notes should not sit between the "
            "mandatory questions and the plain-language explanation.\n\n"
            "## Объяснение простым языком",
            1,
        ),
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "## Три обязательных вопроса must directly precede "
        "## Объяснение простым языком without another markdown heading "
        "between them"
        in errors
    )


def test_cycle_report_validator_rejects_top_level_section_after_plain_explanation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT + "\n## Appendix\n\nExtra notes should not follow the plain explanation.\n",
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "## Объяснение простым языком must be the final top-level report section" in errors


def test_cycle_report_validator_rejects_subsection_after_readiness_scores(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT + "\n### Appendix\n\nExtra notes should not follow readiness scores.\n",
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "### Общая готовность направления must be the final "
        "plain-language subsection"
        in errors
    )


def test_cycle_report_validator_rejects_text_after_readiness_scores(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT + "\nExtra notes should not follow readiness scores.\n",
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert (
        "### Общая готовность направления readiness scores must be the final "
        "plain-language content"
        in errors
    )
