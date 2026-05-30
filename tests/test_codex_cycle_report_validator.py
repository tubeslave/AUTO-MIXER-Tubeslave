import importlib.util
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
    report_path.write_text(body, encoding="utf-8")
    return report_path


VALID_REPORT = """# Automixer Codex Cycle 20260530T000000Z

## Scope

External Paperclip/PiperClip was not used as an agent runner.
No neighboring repositories or external local projects were analyzed, modified,
or used.
Repo-local `.paperclip/reports/` was used only as an Automixer report directory.

Live safety status: no live OSC/WING writes were executed.
No real console was contacted.
No `set_fader`, `set_gain`, `set_eq`, `set_compressor`, `route_output`,
`load_snap`, `save_snap`, `reset`, or `mute` command was run.
No commit, push, reset, destructive cleanup, secret access, or user-file deletion
was performed.

## What Was Found

Repo state was inspected.

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

Проект остается supervised-first.

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


def test_cycle_report_validator_accepts_plan_selected_heading(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT.replace("## Short Plan", "## Plan Selected"),
    )

    assert validator.validate_report(report_path, project_root=tmp_path) == []


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


def test_cycle_report_validator_rejects_top_level_section_after_plain_explanation(tmp_path):
    validator = load_validator_module()
    report_path = write_report(
        tmp_path,
        "automixer_codex_cycle_20260530T000000Z.md",
        VALID_REPORT + "\n## Appendix\n\nExtra notes should not follow the plain explanation.\n",
    )

    errors = validator.validate_report(report_path, project_root=tmp_path)

    assert "## Объяснение простым языком must be the final top-level report section" in errors
