#!/usr/bin/env python3
"""Validate the required shape of a repo-local Codex cycle report."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME_PATTERN = re.compile(r"^automixer_codex_cycle_\d{8}T\d{6}Z\.md$")
REPORT_TITLE_PATTERN = re.compile(
    r"^#\s+Automixer Codex Cycle\s+(\d{8}T\d{6}Z)\s*$",
    re.MULTILINE,
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]\n]+\]\(([^)\n]+)\)")

CODEX_CLI_AGENT_WORKFLOW_PHRASE = "Codex CLI was the only agent workflow used."
EXTERNAL_AGENT_RUNNER_PHRASE = (
    "External Paperclip/PiperClip was not used as an agent runner."
)
REPO_LOCAL_REPORT_PATH_LABEL = "Repo-local report path:"
EXACT_SCOPE_LINE_REQUIREMENTS = (
    (
        CODEX_CLI_AGENT_WORKFLOW_PHRASE,
        "Scope must contain exact agent workflow line: "
        f"{CODEX_CLI_AGENT_WORKFLOW_PHRASE}",
    ),
    (
        EXTERNAL_AGENT_RUNNER_PHRASE,
        "Scope must contain exact external agent boundary line: "
        f"{EXTERNAL_AGENT_RUNNER_PHRASE}",
    ),
)

REQUIRED_PHRASES = (
    "## Scope",
    "## What Was Found",
    "## What Was Done",
    "## Tests And Checks",
    "## What Remains",
    "## Risks",
    "## Changed Files",
    "Live safety status:",
    "no live OSC/WING writes were executed",
    CODEX_CLI_AGENT_WORKFLOW_PHRASE,
    EXTERNAL_AGENT_RUNNER_PHRASE,
    "No neighboring repositories or external local projects were analyzed, modified, or used.",
    "Repo-local `.paperclip/reports/` was used only as an Automixer report directory.",
    REPO_LOCAL_REPORT_PATH_LABEL,
    "No `.env`, tokens, credentials, API keys, or local secrets were accessed or modified.",
    "## Три обязательных вопроса",
    "1. Что нового проект научился делать?",
    "2. Что сейчас является самым слабым местом?",
    "3. Что даст наибольший прогресс в следующем цикле?",
    "## Memory Lint Result",
    "## Объяснение простым языком",
    "### Что было сделано",
    "### Зачем это нужно проекту",
    "### В каком состоянии проект сейчас",
    "### Какие проблемы были обнаружены",
    "### Что нужно сделать дальше",
    "### Общая готовность направления",
)

REQUIRED_READINESS_SCORES = (
    "Исследование",
    "Архитектура",
    "Реализация",
    "Тестирование",
    "Готовность к реальному концерту",
)

REQUIRED_NONEMPTY_SECTIONS = (
    "## Scope",
    "## What Was Found",
    "## What Was Done",
    "## Changed Files",
    "## Tests And Checks",
    "## Memory Lint Result",
    "## What Remains",
    "## Risks",
)

REQUIRED_MID_SECTION_ORDER = (
    "## Tests And Checks",
    "## Memory Lint Result",
    "## What Remains",
)

REQUIRED_DISCOVERY_EVIDENCE = (
    ("branch", ("Branch:", "Current branch", "текущая ветка")),
    ("worktree state", ("git status", "worktree", "рабоч")),
    ("modified files", ("modified", "changed files", "измененные", "изменённые")),
    ("new/untracked files", ("untracked", "new files", "новые файлы")),
    ("repo-local paperclip reports", (".paperclip/reports",)),
    ("Docs reports", ("Docs/reports",)),
    ("test artifacts", ("artifacts", "sessions", "test artifact", "тестовые артефакты")),
    ("TODO/FIXME scan", ("TODO/FIXME",)),
    ("local project tasks", (".ai/", ".ai/project-layer/current_tasks.md", "project issues", "локальные project issues")),
)

DISCOVERY_RECENCY_TERMS = (
    "latest",
    "recent",
    "newest",
    "last",
    "послед",
    "свеж",
)

REQUIRED_RECENT_DISCOVERY_EVIDENCE = (
    ("repo-local paperclip reports", (".paperclip/reports",)),
    ("Docs reports", ("Docs/reports",)),
    ("test artifacts", ("test artifacts", "тестовые артефакты", "artifacts", "sessions")),
)

REQUIRED_PROJECT_GOAL_EVIDENCE = (
    ("README.md", ("README.md",)),
    ("AGENTS.md", ("AGENTS.md",)),
    ("CLAUDE.md", ("CLAUDE.md",)),
    (".ai/memory/project.md", (".ai/memory/project.md",)),
    ("Docs/PROJECT_IMPLEMENTATION_PLAN.md", ("Docs/PROJECT_IMPLEMENTATION_PLAN.md",)),
    ("supervised gate", ("supervised gate", "supervised gate", "супервиз")),
    (
        "read-only/proposal-only/shadow-only/offline-only boundary",
        (
            "read-only",
            "proposal-only",
            "shadow-only",
            "offline-only",
        ),
    ),
)

REQUIRED_SCOPE_PHRASES = (
    CODEX_CLI_AGENT_WORKFLOW_PHRASE,
    EXTERNAL_AGENT_RUNNER_PHRASE,
    "No neighboring repositories or external local projects were analyzed, modified, or used.",
    "Repo-local `.paperclip/reports/` was used only as an Automixer report directory.",
    REPO_LOCAL_REPORT_PATH_LABEL,
    "Live safety status:",
    "no live OSC/WING writes were executed",
    "No `.env`, tokens, credentials, API keys, or local secrets were accessed or modified.",
)

PROJECT_BOUNDARY_AUDIT_TERMS = (
    "/users/dmitrijvolkov/paperclip",
    "/users/dmitrijvolkov/automixer-kimi-paperclip-analyst",
    "paperclip/piperclip",
    "piperclip",
    "external paperclip/piperclip",
    "external paperclip",
    "external piperclip",
    "neighboring repositories",
    "external local projects",
)
PROJECT_BOUNDARY_AUDIT_VERBS = (
    "analyzed",
    "analysed",
    "modified",
    "used",
    "ran",
    "read",
    "opened",
    "inspected",
    "scanned",
    "listed",
    "checked",
    "launched",
    "invoked",
    "was analyzed",
    "were analyzed",
    "was analysed",
    "were analysed",
    "was modified",
    "were modified",
    "was used",
    "were used",
    "was run",
    "were run",
    "was read",
    "were read",
    "was opened",
    "were opened",
    "was inspected",
    "were inspected",
    "was scanned",
    "were scanned",
    "was listed",
    "were listed",
    "was checked",
    "were checked",
    "was launched",
    "were launched",
    "was invoked",
    "were invoked",
)

SECRET_BOUNDARY_AUDIT_TERMS = (
    ".env",
    "api key",
    "api keys",
    "credentials",
    "local secret",
    "local secrets",
    "secret file",
    "secret files",
    "secrets",
)
SECRET_BOUNDARY_AUDIT_VERBS = (
    "accessed",
    "modified",
    "read",
    "opened",
    "inspected",
    "checked",
    "touched",
    "used",
    "was accessed",
    "were accessed",
    "was modified",
    "were modified",
    "was read",
    "were read",
    "was opened",
    "were opened",
    "was inspected",
    "were inspected",
    "was checked",
    "were checked",
    "was touched",
    "were touched",
    "was used",
    "were used",
)
SECRET_BOUNDARY_REQUIRED_SENTENCE_PATTERN = re.compile(
    r"No\s+`?\.env`?,\s+tokens,\s+credentials,\s+API\s+keys,\s+or\s+"
    r"local\s+secrets\s+were\s+accessed\s+or\s+modified\.?",
    re.IGNORECASE,
)

TODO_SCAN_FOCUS_PHRASES = (
    "owned-code",
    "owned code",
    "owned код",
    "owned",
)

TODO_SCAN_EXCLUSION_PHRASES = (
    "excluding",
    "excluded",
    "excluded scan noise",
    "vendored",
    "generated",
    "reference noise",
    "scan noise",
    "исключ",
)
TODO_SCAN_COMMAND_PHRASES = (
    "rg --hidden --glob",
)

PLAN_SECTION_HEADINGS = (
    "## Short Plan",
    "## Plan Selected",
)
MAX_PLAN_ITEMS = 3

REQUIRED_CORE_SECTION_ORDER = (
    ("## Scope", ("## Scope",)),
    ("## What Was Found", ("## What Was Found",)),
    ("## Short Plan or ## Plan Selected", PLAN_SECTION_HEADINGS),
    ("## What Was Done", ("## What Was Done",)),
    ("## Changed Files", ("## Changed Files",)),
    ("## Tests And Checks", ("## Tests And Checks",)),
    ("## Memory Lint Result", ("## Memory Lint Result",)),
    ("## What Remains", ("## What Remains",)),
    ("## Risks", ("## Risks",)),
)

TEST_RESULT_EVIDENCE_PATTERN = re.compile(
    r"\b(pass(?:ed)?|fail(?:ed)?|skipped|blocked|not run)\b"
    r"|не запуск|не удалось|прош|ошиб",
    re.IGNORECASE,
)
MEMORY_LINT_EVIDENCE_TERMS = (
    "memory lint",
    "memory_lint.py",
    "tools/memory/memory_lint.py",
)
MEMORY_LINT_REQUIRED_COMMAND = "python3 tools/memory/memory_lint.py"
MEMORY_LINT_FORBIDDEN_ARGUMENTS = ("--root",)
MEMORY_LINT_STATUS_PATTERN = re.compile(r"\b(?:PASS|FAIL)\b", re.IGNORECASE)

MANDATORY_QUESTIONS_HEADING = "## Три обязательных вопроса"
MANDATORY_QUESTIONS = (
    "1. Что нового проект научился делать?",
    "2. Что сейчас является самым слабым местом?",
    "3. Что даст наибольший прогресс в следующем цикле?",
)

REQUIRED_LIVE_SAFETY_PHRASES = (
    "No real console was contacted",
)
REAL_CONSOLE_CONTACT_AUDIT_TERMS = (
    "real console",
)
REAL_CONSOLE_CONTACT_AUDIT_VERBS = (
    "was contacted",
    "were contacted",
    "contacted",
    "connected to",
    "was connected",
    "were connected",
)

LIVE_WRITE_AUDIT_TERMS = (
    "live osc/wing writes",
)
LIVE_WRITE_AUDIT_VERBS = (
    "was executed",
    "were executed",
    "was performed",
    "were performed",
    "was sent",
    "were sent",
    "was run",
    "were run",
    "was applied",
    "were applied",
)

NON_DESTRUCTIVE_ACTION_AUDIT_TERMS = (
    "commit",
    "push",
    "reset",
    "destructive cleanup",
    "secret access",
    "user-file deletion",
)
NON_DESTRUCTIVE_ACTION_AUDIT_VERBS = (
    "was performed",
    "were performed",
    "не выполня",
    "не удал",
)

FORBIDDEN_LIVE_COMMANDS = (
    "set_fader",
    "set_gain",
    "set_trim",
    "set_input_trim",
    "set_channel_trim",
    "set_preamp_trim",
    "set_preamp_gain",
    "set_eq",
    "set_compressor",
    "route_output",
    "load_snap",
    "save_snap",
    "reset",
    "mute",
    "unmute",
    "set_insert",
    "set_fx_model",
    "set_fx_on",
    "set_fx_mix",
    "set_fx_parameter",
)
FORBIDDEN_LIVE_COMMAND_CONTRADICTION_TERMS = FORBIDDEN_LIVE_COMMANDS
NEGATED_COMMAND_AUDIT_VERBS = (
    "command was run",
    "commands were run",
    "command was sent",
    "commands were sent",
    "was run",
    "were run",
    "was sent",
    "were sent",
    "was executed",
    "were executed",
    "was applied",
    "were applied",
    "не выполня",
    "не отправ",
)
COMMAND_AUDIT_CONTEXT_PATTERN = re.compile(
    r"\b(?:commands?|run|runs|ran|send|sent|execute|executed|apply|applied)\b"
)

NEXT_STEPS_HEADING = "### Что нужно сделать дальше"
PLAIN_EXPLANATION_HEADING = "## Объяснение простым языком"
MAX_NEXT_STEP_ITEMS = 5

REQUIRED_END_SECTION_ORDER = (
    MANDATORY_QUESTIONS_HEADING,
    *MANDATORY_QUESTIONS,
    "## Объяснение простым языком",
    "### Что было сделано",
    "### Зачем это нужно проекту",
    "### В каком состоянии проект сейчас",
    "### Какие проблемы были обнаружены",
    "### Что нужно сделать дальше",
    "### Общая готовность направления",
)

REQUIRED_PLAIN_EXPLANATION_SUBSECTIONS = (
    "### Что было сделано",
    "### Зачем это нужно проекту",
    "### В каком состоянии проект сейчас",
    "### Какие проблемы были обнаружены",
    "### Что нужно сделать дальше",
    "### Общая готовность направления",
)

PROJECT_STATE_EXPLANATION_HEADING = "### В каком состоянии проект сейчас"
READINESS_EXPLANATION_HEADING = "### Общая готовность направления"
REQUIRED_PROJECT_STATE_EVIDENCE = (
    ("what already works", ("уже работает", "работает", "already works", "works")),
    ("what works partially", ("частично", "partially")),
    (
        "what is not implemented",
        ("еще не реализ", "ещё не реализ", "не реализ", "not implemented", "not yet implemented"),
    ),
    ("risks", ("риск", "risk")),
)
REQUIRED_GIT_STATUS_COMMAND = "git status --short"
BRANCH_EVIDENCE_PATTERN = re.compile(
    r"^[ \t]*(?:Branch|Current branch|Текущая ветка)[ \t]*:[ \t]*(.*?)[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_readiness_scores(text: str) -> list[str]:
    errors: list[str] = []
    section_body = _extract_heading_section(text, READINESS_EXPLANATION_HEADING)

    for label in REQUIRED_READINESS_SCORES:
        pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(\d{{1,3}})\s*%\s*$",
            re.MULTILINE,
        )
        match = pattern.search(section_body)
        if not match:
            errors.append(f"missing readiness score: {label}: N%")
            continue

        value = int(match.group(1))
        if value > 100:
            errors.append(f"readiness score must be between 0 and 100: {label}: {value}%")

    return errors


def _validate_readiness_scores_are_final_content(text: str) -> list[str]:
    section_body = _extract_heading_section(text, READINESS_EXPLANATION_HEADING)
    if not section_body:
        return []

    score_matches = []
    for label in REQUIRED_READINESS_SCORES:
        pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*\d{{1,3}}\s*%\s*$",
            re.MULTILINE,
        )
        match = pattern.search(section_body)
        if not match:
            return []

        score_matches.append(match)

    final_score = max(score_matches, key=lambda match: match.end())
    if section_body[final_score.end():].strip():
        return [
            f"{READINESS_EXPLANATION_HEADING} readiness scores must be the "
            "final plain-language content"
        ]

    return []


def _validate_report_title(text: str, report_path: Path) -> list[str]:
    match = REPORT_TITLE_PATTERN.search(text)
    if not match:
        return ["missing report title: # Automixer Codex Cycle YYYYMMDDTHHMMSSZ"]

    expected_timestamp = report_path.stem.removeprefix("automixer_codex_cycle_")
    actual_timestamp = match.group(1)
    if actual_timestamp != expected_timestamp:
        return [
            "report title timestamp must match filename: "
            f"{actual_timestamp} != {expected_timestamp}"
        ]

    return []


def _validate_report_filename_timestamp(
    report_path: Path,
    *,
    now: datetime | None = None,
) -> list[str]:
    timestamp = report_path.stem.removeprefix("automixer_codex_cycle_")
    if timestamp == report_path.stem:
        return []

    try:
        parsed = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return [f"report filename timestamp must be a valid UTC timestamp: {timestamp}"]

    if now is not None:
        comparison_now = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        if parsed > comparison_now.astimezone(timezone.utc):
            return [f"report filename timestamp must not be in the future: {timestamp}"]

    return []


def _extract_plan_items(text: str, heading: str) -> list[str]:
    section_body = _extract_heading_section(text, heading)
    if not section_body:
        return []

    return [
        line
        for line in section_body.splitlines()
        if re.match(r"^\s*(?:[-*+]\s+|\d+[\.)]\s+)", line)
    ]


def _validate_plan_section(text: str) -> list[str]:
    for heading in PLAN_SECTION_HEADINGS:
        section_body = _extract_heading_section(text, heading)
        if re.search(r"[0-9A-Za-zА-Яа-яЁё]", section_body):
            return []

    return [
        "missing required short plan section: "
        + " or ".join(PLAN_SECTION_HEADINGS)
    ]


def _validate_plan_item_limit(text: str) -> list[str]:
    for heading in PLAN_SECTION_HEADINGS:
        if not _extract_heading_section(text, heading):
            continue

        plan_items = _extract_plan_items(text, heading)
        if not plan_items:
            return [
                f"{heading} must contain between 1 and {MAX_PLAN_ITEMS} list items"
            ]

        if len(plan_items) > MAX_PLAN_ITEMS:
            return [
                f"{heading} must contain no more than {MAX_PLAN_ITEMS} list items"
            ]

    return []


def _validate_required_sections_have_content(text: str) -> list[str]:
    errors: list[str] = []

    for heading in REQUIRED_NONEMPTY_SECTIONS:
        section_body = _extract_heading_section(text, heading)
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]", section_body):
            errors.append(f"required report section has no content: {heading}")

    return errors


def _validate_scope_evidence(text: str) -> list[str]:
    scope_body = _extract_heading_section(text, "## Scope")
    if not scope_body:
        return []

    errors: list[str] = []
    for phrase in REQUIRED_SCOPE_PHRASES:
        if not _contains_required_phrase(scope_body, phrase):
            errors.append(f"missing required Scope evidence: {phrase}")

    scope_lines = {line.strip() for line in scope_body.splitlines()}
    for phrase, error in EXACT_SCOPE_LINE_REQUIREMENTS:
        if phrase not in scope_lines:
            errors.append(error)

    return errors


def _validate_scope_repo_local_report_path(
    text: str,
    project_root: Path,
) -> list[str]:
    scope_body = _extract_heading_section(text, "## Scope")
    if not scope_body:
        return []

    expected_path = str((project_root / ".paperclip" / "reports").resolve())
    labeled_path_pattern = re.compile(
        rf"{re.escape(REPO_LOCAL_REPORT_PATH_LABEL)}\s*`?"
        rf"{re.escape(expected_path)}`?\.?"
    )
    if labeled_path_pattern.search(scope_body):
        return []

    errors = []
    if expected_path not in scope_body:
        errors.append(
            "Scope must include repo-local report path under Automixer project: "
            f"{expected_path}"
        )

    errors.append(
        "Scope must attach repo-local report path to label: "
        f"{REPO_LOCAL_REPORT_PATH_LABEL} {expected_path}"
    )
    return errors


def _validate_project_boundary_contradictions(text: str) -> list[str]:
    report_body_without_scope = _remove_heading_section(text, "## Scope")
    normalized_report_body = _collapse_whitespace(report_body_without_scope).lower()

    contradicted_terms = [
        term
        for term in PROJECT_BOUNDARY_AUDIT_TERMS
        if term in normalized_report_body
        and _has_affirmed_action_audit(
            normalized_report_body,
            term,
            PROJECT_BOUNDARY_AUDIT_VERBS,
        )
    ]
    if not contradicted_terms:
        return []

    return [
        "report body contradicts repo-local project boundary evidence: "
        + ", ".join(contradicted_terms)
    ]


def _validate_scope_project_boundary_contradictions(text: str) -> list[str]:
    scope_body = _extract_heading_section(text, "## Scope")
    if not scope_body:
        return []

    normalized_scope = _collapse_whitespace(scope_body).lower()
    contradicted_terms = [
        term
        for term in PROJECT_BOUNDARY_AUDIT_TERMS
        if term in normalized_scope
        and _has_affirmed_action_audit(
            normalized_scope,
            term,
            PROJECT_BOUNDARY_AUDIT_VERBS,
        )
    ]
    if not contradicted_terms:
        return []

    return [
        "Scope contradicts repo-local project boundary evidence: "
        + ", ".join(contradicted_terms)
    ]


def _find_secret_boundary_contradictions(section_text: str) -> list[str]:
    section_without_required_audit = SECRET_BOUNDARY_REQUIRED_SENTENCE_PATTERN.sub(
        "",
        section_text,
    )
    normalized_section = _collapse_whitespace(section_without_required_audit).lower()
    if not normalized_section:
        return []

    return [
        term
        for term in SECRET_BOUNDARY_AUDIT_TERMS
        if term in normalized_section
        and _has_affirmed_action_audit(
            normalized_section,
            term,
            SECRET_BOUNDARY_AUDIT_VERBS,
        )
    ]


def _validate_secret_boundary_contradictions(text: str) -> list[str]:
    errors: list[str] = []

    scope_terms = _find_secret_boundary_contradictions(
        _extract_heading_section(text, "## Scope")
    )
    if scope_terms:
        errors.append(
            "Scope contradicts secret boundary evidence: "
            + ", ".join(scope_terms)
        )

    body_terms = _find_secret_boundary_contradictions(
        _remove_heading_section(text, "## Scope")
    )
    if body_terms:
        errors.append(
            "report body contradicts secret boundary evidence: "
            + ", ".join(body_terms)
        )

    return errors


def _validate_discovery_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## What Was Found")
    if not section_body:
        return []

    normalized_section = _collapse_whitespace(section_body).lower()
    errors: list[str] = []
    for label, phrases in REQUIRED_DISCOVERY_EVIDENCE:
        if not any(phrase.lower() in normalized_section for phrase in phrases):
            errors.append(f"What Was Found is missing discovery evidence: {label}")

    return errors


def _validate_git_status_and_branch_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## What Was Found")
    if not section_body:
        return []

    errors: list[str] = []
    normalized_section = _collapse_whitespace(section_body).lower()
    if REQUIRED_GIT_STATUS_COMMAND not in normalized_section:
        errors.append(
            "What Was Found must include exact worktree command evidence: "
            f"{REQUIRED_GIT_STATUS_COMMAND}"
        )

    branch_match = BRANCH_EVIDENCE_PATTERN.search(section_body)
    branch_value = branch_match.group(1).strip("` .") if branch_match else ""
    if not branch_value or not re.search(r"[0-9A-Za-zА-Яа-яЁё/_\-.]", branch_value):
        errors.append(
            "What Was Found must include current branch evidence, "
            "e.g. Branch: `branch-name`"
        )

    return errors


def _validate_recent_discovery_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## What Was Found")
    if not section_body:
        return []

    normalized_lines = [
        _collapse_whitespace(line).lower()
        for line in section_body.splitlines()
    ]
    errors: list[str] = []
    for label, phrases in REQUIRED_RECENT_DISCOVERY_EVIDENCE:
        has_recent_line = any(
            any(phrase.lower() in line for phrase in phrases)
            and any(term in line for term in DISCOVERY_RECENCY_TERMS)
            for line in normalized_lines
        )
        if not has_recent_line:
            errors.append(
                f"What Was Found must mention latest/recent discovery evidence: {label}"
            )

    return errors


def _validate_todo_scan_boundary_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## What Was Found")
    if not section_body:
        return []

    normalized_section = _collapse_whitespace(section_body).lower()
    if "todo/fixme" not in normalized_section:
        return []

    errors: list[str] = []
    if not any(phrase in normalized_section for phrase in TODO_SCAN_FOCUS_PHRASES):
        errors.append(
            "What Was Found TODO/FIXME scan must mention owned-code focus"
        )

    if not any(phrase in normalized_section for phrase in TODO_SCAN_EXCLUSION_PHRASES):
        errors.append(
            "What Was Found TODO/FIXME scan must mention excluded "
            "vendored/generated/reference noise"
        )

    if not any(phrase in normalized_section for phrase in TODO_SCAN_COMMAND_PHRASES):
        errors.append(
            "What Was Found TODO/FIXME scan must mention rg --hidden --glob "
            "command pattern"
        )

    return errors


def _validate_project_goal_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## What Was Found")
    if not section_body:
        return []

    normalized_section = _collapse_whitespace(section_body).lower()
    errors: list[str] = []
    for label, phrases in REQUIRED_PROJECT_GOAL_EVIDENCE:
        if not any(phrase.lower() in normalized_section for phrase in phrases):
            errors.append(f"What Was Found is missing project-goal evidence: {label}")

    return errors


def _validate_tests_and_checks_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## Tests And Checks")
    if not section_body:
        return []

    if TEST_RESULT_EVIDENCE_PATTERN.search(section_body):
        return []

    return [
        "Tests And Checks must include outcome evidence: passed, failed, "
        "skipped, blocked, or not run"
    ]


def _validate_memory_lint_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## Tests And Checks")
    if not section_body:
        return []

    lines = section_body.splitlines()
    evidence_chunks: list[str] = []
    for index, line in enumerate(lines):
        normalized_line = line.lower()
        if not any(term in normalized_line for term in MEMORY_LINT_EVIDENCE_TERMS):
            continue

        evidence_chunks.append("\n".join(lines[index:index + 3]))

    if not evidence_chunks:
        return ["Tests And Checks must include memory lint result"]

    forbidden_arguments = [
        argument
        for argument in MEMORY_LINT_FORBIDDEN_ARGUMENTS
        if any(argument in chunk for chunk in evidence_chunks)
    ]
    if forbidden_arguments:
        return [
            "Tests And Checks memory lint result must use repo-local memory "
            "lint without alternate-root arguments: "
            + ", ".join(forbidden_arguments)
        ]

    if any(TEST_RESULT_EVIDENCE_PATTERN.search(chunk) for chunk in evidence_chunks):
        return []

    return [
        "Tests And Checks memory lint result must include outcome evidence: "
        "passed, failed, skipped, blocked, or not run"
    ]


def _validate_memory_lint_result_section(text: str) -> list[str]:
    section_body = _extract_heading_section(text, "## Memory Lint Result")
    if not section_body:
        return []

    errors: list[str] = []
    normalized_section = _collapse_whitespace(section_body).lower()
    if not any(term in normalized_section for term in MEMORY_LINT_EVIDENCE_TERMS):
        errors.append("Memory Lint Result must mention memory lint or memory_lint.py")

    if not _contains_required_phrase(section_body, MEMORY_LINT_REQUIRED_COMMAND):
        errors.append(
            "Memory Lint Result must include exact command: "
            f"{MEMORY_LINT_REQUIRED_COMMAND}"
        )

    forbidden_arguments = [
        argument
        for argument in MEMORY_LINT_FORBIDDEN_ARGUMENTS
        if argument in section_body
    ]
    if forbidden_arguments:
        errors.append(
            "Memory Lint Result must use repo-local memory lint without "
            "alternate-root arguments: "
            + ", ".join(forbidden_arguments)
        )

    if not TEST_RESULT_EVIDENCE_PATTERN.search(section_body):
        errors.append(
            "Memory Lint Result must include outcome evidence: passed, failed, "
            "skipped, blocked, or not run"
        )

    if not MEMORY_LINT_STATUS_PATTERN.search(section_body):
        errors.append("Memory Lint Result must include explicit PASS or FAIL status")

    return errors


def _validate_ordered_end_sections(text: str) -> list[str]:
    errors: list[str] = []
    previous_phrase = ""
    search_start = 0

    for phrase in REQUIRED_END_SECTION_ORDER:
        current_index = text.find(phrase, search_start)
        if current_index == -1:
            if phrase in text and previous_phrase:
                errors.append(
                    "required end sections are out of order: "
                    f"{previous_phrase} must appear before {phrase}"
                )
                break
            continue
        previous_phrase = phrase
        search_start = current_index + len(phrase)

    return errors


def _validate_mandatory_questions_directly_precede_plain_explanation(
    text: str,
) -> list[str]:
    questions_match = re.search(
        rf"^{re.escape(MANDATORY_QUESTIONS_HEADING)}\s*$",
        text,
        re.MULTILINE,
    )
    plain_match = re.search(
        rf"^{re.escape(PLAIN_EXPLANATION_HEADING)}\s*$",
        text,
        re.MULTILINE,
    )
    if not questions_match or not plain_match:
        return []

    if plain_match.start() < questions_match.end():
        return []

    intervening_heading = re.search(
        r"^#{1,6}\s+",
        text[questions_match.end():plain_match.start()],
        re.MULTILINE,
    )
    if intervening_heading:
        return [
            f"{MANDATORY_QUESTIONS_HEADING} must directly precede "
            f"{PLAIN_EXPLANATION_HEADING} without another markdown heading "
            "between them"
        ]

    return []


def _validate_ordered_mid_sections(text: str) -> list[str]:
    errors: list[str] = []
    previous_heading = ""
    search_start = 0

    for heading in REQUIRED_MID_SECTION_ORDER:
        current_index = text.find(heading, search_start)
        if current_index == -1:
            if heading in text and previous_heading:
                errors.append(
                    "required middle sections are out of order: "
                    f"{previous_heading} must appear before {heading}"
                )
                break
            continue
        previous_heading = heading
        search_start = current_index + len(heading)

    return errors


def _validate_ordered_core_sections(text: str) -> list[str]:
    errors: list[str] = []
    previous_label = ""
    previous_index = -1

    for label, heading_options in REQUIRED_CORE_SECTION_ORDER:
        heading_indexes = [
            heading_index
            for heading in heading_options
            if (heading_index := _heading_index(text, heading)) != -1
        ]
        if not heading_indexes:
            continue

        current_index = min(heading_indexes)
        if current_index < previous_index:
            errors.append(
                "required core sections are out of order: "
                f"{previous_label} must appear before {label}"
            )
            break

        previous_label = label
        previous_index = current_index

    return errors


def _extract_markdown_section(text: str, heading: str) -> str:
    heading_pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    heading_match = heading_pattern.search(text)
    if not heading_match:
        return ""

    next_heading = re.search(r"^##\s+", text[heading_match.end():], re.MULTILINE)
    section_end = (
        heading_match.end() + next_heading.start()
        if next_heading
        else len(text)
    )
    return text[heading_match.end():section_end].strip()


def _normalize_markdown_link_target(target: str) -> str:
    target = target.strip()
    if target.startswith("<") and ">" in target:
        normalized = target[1:target.index(">")]
    else:
        normalized = target.split()[0] if target else target

    return _strip_local_file_location(normalized)


def _strip_local_file_location(target: str) -> str:
    target = re.sub(r"#L?\d+(?:-L?\d+)?$", "", target)
    return re.sub(r":\d+$", "", target)


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_required_phrase(text: str, phrase: str) -> bool:
    if phrase.isascii():
        return _collapse_whitespace(phrase).lower() in _collapse_whitespace(text).lower()

    return _collapse_whitespace(phrase) in _collapse_whitespace(text)


def _validate_changed_file_links(
    text: str,
    report_path: Path,
    project_root: Path,
) -> list[str]:
    errors: list[str] = []
    section_body = _extract_markdown_section(text, "## Changed Files")
    if not section_body:
        return errors

    repo_local_link_found = False
    self_report_link_found = False
    resolved_project_root = project_root.resolve()
    resolved_report_path = report_path.resolve()
    for raw_target in MARKDOWN_LINK_PATTERN.findall(section_body):
        target = _normalize_markdown_link_target(raw_target)
        if not target or target.startswith("#") or "://" in target:
            continue

        linked_path = Path(target)
        if not linked_path.is_absolute():
            linked_path = report_path.parent / linked_path

        resolved_linked_path = linked_path.resolve()
        if not _is_relative_to(resolved_linked_path, resolved_project_root):
            errors.append(f"changed file link points outside project: {target}")
            continue

        if not linked_path.is_file():
            errors.append(f"changed file link does not exist: {target}")
            continue

        repo_local_link_found = True
        if resolved_linked_path == resolved_report_path:
            self_report_link_found = True

    if not repo_local_link_found:
        errors.append(
            "Changed Files section must include at least one markdown link "
            "to a repo-local file"
        )

    if not self_report_link_found:
        errors.append(
            "Changed Files section must include a markdown link to this cycle report"
        )

    return errors


def _extract_heading_section(text: str, heading: str) -> str:
    heading_pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    heading_match = heading_pattern.search(text)
    if not heading_match:
        return ""

    heading_level = len(heading) - len(heading.lstrip("#"))
    remaining_text = text[heading_match.end():]
    next_heading = re.search(rf"^#{{1,{heading_level}}}\s+", remaining_text, re.MULTILINE)
    section_end = (
        heading_match.end() + next_heading.start()
        if next_heading
        else len(text)
    )
    return text[heading_match.end():section_end].strip()


def _heading_index(text: str, heading: str) -> int:
    heading_match = re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)
    return heading_match.start() if heading_match else -1


def _validate_next_steps_limit(text: str) -> list[str]:
    section_body = _extract_heading_section(text, NEXT_STEPS_HEADING)
    if not section_body:
        return []

    step_items = [
        line
        for line in section_body.splitlines()
        if re.match(r"^\s*(?:[-*+]\s+|\d+[\.)]\s+)", line)
    ]
    if not step_items:
        return [
            f"{NEXT_STEPS_HEADING} must contain between 1 and "
            f"{MAX_NEXT_STEP_ITEMS} list items"
        ]

    if len(step_items) > MAX_NEXT_STEP_ITEMS:
        return [
            f"{NEXT_STEPS_HEADING} must contain no more than "
            f"{MAX_NEXT_STEP_ITEMS} list items"
        ]

    return []


def _validate_plain_explanation_is_final_major_section(text: str) -> list[str]:
    heading_pattern = re.compile(
        rf"^{re.escape(PLAIN_EXPLANATION_HEADING)}\s*$",
        re.MULTILINE,
    )
    heading_match = heading_pattern.search(text)
    if not heading_match:
        return []

    later_top_level_heading = re.search(
        r"^##\s+",
        text[heading_match.end():],
        re.MULTILINE,
    )
    if later_top_level_heading:
        return [
            f"{PLAIN_EXPLANATION_HEADING} must be the final top-level report section"
        ]

    return []


def _validate_readiness_is_final_plain_subsection(text: str) -> list[str]:
    heading_pattern = re.compile(
        rf"^{re.escape(READINESS_EXPLANATION_HEADING)}\s*$",
        re.MULTILINE,
    )
    heading_match = heading_pattern.search(text)
    if not heading_match:
        return []

    later_plain_subsection = re.search(
        r"^###\s+",
        text[heading_match.end():],
        re.MULTILINE,
    )
    if later_plain_subsection:
        return [
            f"{READINESS_EXPLANATION_HEADING} must be the final "
            "plain-language subsection"
        ]

    return []


def _validate_plain_explanation_subsections_have_content(text: str) -> list[str]:
    if not _extract_heading_section(text, PLAIN_EXPLANATION_HEADING):
        return []

    errors: list[str] = []
    for heading in REQUIRED_PLAIN_EXPLANATION_SUBSECTIONS:
        section_body = _extract_heading_section(text, heading)
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]", section_body):
            errors.append(f"plain-language subsection has no content: {heading}")

    return errors


def _validate_project_state_explanation_evidence(text: str) -> list[str]:
    section_body = _extract_heading_section(text, PROJECT_STATE_EXPLANATION_HEADING)
    if not section_body:
        return []

    normalized_section = _collapse_whitespace(section_body).lower()
    errors: list[str] = []
    for label, phrases in REQUIRED_PROJECT_STATE_EVIDENCE:
        if not any(phrase in normalized_section for phrase in phrases):
            errors.append(
                f"plain-language state subsection must mention {label}"
            )

    return errors


def _validate_mandatory_question_answers(text: str) -> list[str]:
    section_body = _extract_heading_section(text, MANDATORY_QUESTIONS_HEADING)
    if not section_body:
        return []

    question_rows: list[tuple[str, int, int]] = []
    search_start = 0
    for question in MANDATORY_QUESTIONS:
        question_index = section_body.find(question, search_start)
        if question_index == -1:
            continue

        line_start = section_body.rfind("\n", 0, question_index) + 1
        question_rows.append((question, question_index, line_start))
        search_start = question_index + len(question)

    errors: list[str] = []
    for index, (question, question_index, _line_start) in enumerate(question_rows):
        answer_start = question_index + len(question)
        answer_end = (
            question_rows[index + 1][2]
            if index + 1 < len(question_rows)
            else len(section_body)
        )
        answer_body = section_body[answer_start:answer_end].strip()
        if not re.search(r"[0-9A-Za-zА-Яа-яЁё]", answer_body):
            errors.append(f"mandatory question needs an answer: {question}")

    return errors


def _validate_live_safety_evidence(text: str) -> list[str]:
    errors: list[str] = []
    audit_section = _extract_heading_section(text, "## Scope") or text
    lower_text = audit_section.lower()
    normalized_lower_text = _collapse_whitespace(audit_section).lower()

    for phrase in REQUIRED_LIVE_SAFETY_PHRASES:
        if phrase.lower() not in normalized_lower_text:
            errors.append(f"missing required live-safety evidence: {phrase}")

    contradicted_real_console_contact_terms = [
        term
        for term in REAL_CONSOLE_CONTACT_AUDIT_TERMS
        if term in normalized_lower_text
        and _has_affirmed_action_audit(
            normalized_lower_text,
            term,
            REAL_CONSOLE_CONTACT_AUDIT_VERBS,
        )
    ]
    if contradicted_real_console_contact_terms:
        errors.append(
            "real-console audit contradicts no-contact safety evidence: "
            + ", ".join(contradicted_real_console_contact_terms)
        )

    contradicted_live_write_terms = [
        term
        for term in LIVE_WRITE_AUDIT_TERMS
        if term in normalized_lower_text
        and _has_affirmed_action_audit(
            normalized_lower_text,
            term,
            LIVE_WRITE_AUDIT_VERBS,
        )
    ]
    if contradicted_live_write_terms:
        errors.append(
            "live write audit contradicts no-write safety evidence: "
            + ", ".join(contradicted_live_write_terms)
        )

    missing_commands = [
        command
        for command in FORBIDDEN_LIVE_COMMANDS
        if command.lower() not in lower_text
        or not _has_command_audit_context(normalized_lower_text, command)
    ]
    if missing_commands:
        errors.append(
            "missing forbidden command audit evidence: "
            + ", ".join(missing_commands)
        )

    unnegated_commands = [
        command
        for command in FORBIDDEN_LIVE_COMMANDS
        if command.lower() in lower_text
        and not _has_negated_command_audit(normalized_lower_text, command)
    ]
    if unnegated_commands:
        errors.append(
            "forbidden command audit must explicitly say command was not run: "
            + ", ".join(unnegated_commands)
        )

    contradicted_commands = [
        command
        for command in FORBIDDEN_LIVE_COMMAND_CONTRADICTION_TERMS
        if command.lower() in lower_text
        and _has_affirmed_command_audit(normalized_lower_text, command)
    ]
    if contradicted_commands:
        errors.append(
            "forbidden command audit contradicts no-run safety evidence: "
            + ", ".join(contradicted_commands)
        )

    missing_non_destructive_terms = [
        term
        for term in NON_DESTRUCTIVE_ACTION_AUDIT_TERMS
        if term.lower() not in normalized_lower_text
    ]
    if missing_non_destructive_terms:
        errors.append(
            "missing non-destructive action audit evidence: "
            + ", ".join(missing_non_destructive_terms)
        )

    unnegated_non_destructive_terms = [
        term
        for term in NON_DESTRUCTIVE_ACTION_AUDIT_TERMS
        if term.lower() in normalized_lower_text
        and not _has_negated_action_audit(
            normalized_lower_text,
            term,
            NON_DESTRUCTIVE_ACTION_AUDIT_VERBS,
        )
    ]
    if unnegated_non_destructive_terms:
        errors.append(
            "non-destructive action audit must explicitly say action was not performed: "
            + ", ".join(unnegated_non_destructive_terms)
        )

    contradicted_non_destructive_terms = [
        term
        for term in NON_DESTRUCTIVE_ACTION_AUDIT_TERMS
        if term.lower() in normalized_lower_text
        and _has_affirmed_action_audit(
            normalized_lower_text,
            term,
            NON_DESTRUCTIVE_ACTION_AUDIT_VERBS,
        )
    ]
    if contradicted_non_destructive_terms:
        errors.append(
            "non-destructive action audit contradicts no-action evidence: "
            + ", ".join(contradicted_non_destructive_terms)
        )

    report_body_without_scope = _remove_heading_section(text, "## Scope")
    normalized_report_body = _collapse_whitespace(report_body_without_scope).lower()
    lower_report_body = report_body_without_scope.lower()

    body_contradicted_live_write_terms = [
        term
        for term in LIVE_WRITE_AUDIT_TERMS
        if term in normalized_report_body
        and _has_affirmed_action_audit(
            normalized_report_body,
            term,
            LIVE_WRITE_AUDIT_VERBS,
        )
    ]
    if body_contradicted_live_write_terms:
        errors.append(
            "report body contradicts no-write safety evidence: "
            + ", ".join(body_contradicted_live_write_terms)
        )

    body_contradicted_real_console_contact_terms = [
        term
        for term in REAL_CONSOLE_CONTACT_AUDIT_TERMS
        if term in normalized_report_body
        and _has_affirmed_action_audit(
            normalized_report_body,
            term,
            REAL_CONSOLE_CONTACT_AUDIT_VERBS,
        )
    ]
    if body_contradicted_real_console_contact_terms:
        errors.append(
            "report body contradicts no-contact safety evidence: "
            + ", ".join(body_contradicted_real_console_contact_terms)
        )

    body_contradicted_commands = [
        command
        for command in FORBIDDEN_LIVE_COMMAND_CONTRADICTION_TERMS
        if command.lower() in lower_report_body
        and _has_affirmed_command_audit(normalized_report_body, command)
    ]
    if body_contradicted_commands:
        errors.append(
            "report body contradicts no-run safety evidence: "
            + ", ".join(body_contradicted_commands)
        )

    body_contradicted_non_destructive_terms = [
        term
        for term in NON_DESTRUCTIVE_ACTION_AUDIT_TERMS
        if term.lower() in normalized_report_body
        and _has_affirmed_action_audit(
            normalized_report_body,
            term,
            NON_DESTRUCTIVE_ACTION_AUDIT_VERBS,
        )
    ]
    if body_contradicted_non_destructive_terms:
        errors.append(
            "report body contradicts no-action evidence: "
            + ", ".join(body_contradicted_non_destructive_terms)
        )

    return errors


def _has_negated_command_audit(normalized_lower_text: str, command: str) -> bool:
    return _has_negated_action_audit(
        normalized_lower_text,
        command,
        NEGATED_COMMAND_AUDIT_VERBS,
    )


def _has_command_audit_context(normalized_lower_text: str, command: str) -> bool:
    action_lower = command.lower()
    for match in re.finditer(re.escape(action_lower), normalized_lower_text):
        sentence_start = max(
            normalized_lower_text.rfind(".", 0, match.start()),
            normalized_lower_text.rfind("!", 0, match.start()),
            normalized_lower_text.rfind("?", 0, match.start()),
        ) + 1
        sentence_end_candidates = [
            index
            for index in (
                normalized_lower_text.find(".", match.end()),
                normalized_lower_text.find("!", match.end()),
                normalized_lower_text.find("?", match.end()),
            )
            if index != -1
        ]
        sentence_end = (
            min(sentence_end_candidates)
            if sentence_end_candidates
            else len(normalized_lower_text)
        )
        sentence = normalized_lower_text[sentence_start:sentence_end].strip()
        action_offset = match.start() - sentence_start
        clause_start, clause_end = _action_clause_bounds(sentence, action_offset)
        clause = sentence[clause_start:clause_end].strip()
        if COMMAND_AUDIT_CONTEXT_PATTERN.search(clause):
            return True

    return False


def _has_affirmed_command_audit(normalized_lower_text: str, command: str) -> bool:
    return _has_affirmed_action_audit(
        normalized_lower_text,
        command,
        NEGATED_COMMAND_AUDIT_VERBS,
    )


def _has_negated_action_audit(
    normalized_lower_text: str,
    action: str,
    audit_verbs: Iterable[str],
) -> bool:
    action_lower = action.lower()
    for match in re.finditer(re.escape(action_lower), normalized_lower_text):
        sentence_start = max(
            normalized_lower_text.rfind(".", 0, match.start()),
            normalized_lower_text.rfind("!", 0, match.start()),
            normalized_lower_text.rfind("?", 0, match.start()),
        ) + 1
        sentence_end_candidates = [
            index
            for index in (
                normalized_lower_text.find(".", match.end()),
                normalized_lower_text.find("!", match.end()),
                normalized_lower_text.find("?", match.end()),
            )
            if index != -1
        ]
        sentence_end = (
            min(sentence_end_candidates)
            if sentence_end_candidates
            else len(normalized_lower_text)
        )
        sentence = normalized_lower_text[sentence_start:sentence_end].strip()
        action_offset = match.start() - sentence_start
        clause_start, clause_end = _action_clause_bounds(sentence, action_offset)
        clause = sentence[clause_start:clause_end].strip()
        before_action = sentence[clause_start:action_offset]
        negated_before = re.search(r"\bno\b", before_action) is not None
        negated_russian = "не " in clause
        negated_execution = re.search(
            r"\b(?:was|were)\s+not\s+(?:run|sent|executed|performed|applied|used)\b"
            r"|\bcommands?\s+(?:was|were)\s+not\s+(?:run|sent|executed|applied)\b"
            r"|\b(?:did|do|does)\s+not\s+(?:run|send|execute|perform|apply|use)\b"
            r"|\bnever\s+(?:run|sent|executed|performed|applied|used)\b",
            clause,
        ) is not None
        has_execution_verb = (
            any(verb in clause for verb in audit_verbs)
            or negated_execution
        )
        if (negated_before or negated_russian or negated_execution) and has_execution_verb:
            return True

    return False


def _has_affirmed_action_audit(
    normalized_lower_text: str,
    action: str,
    audit_verbs: Iterable[str],
) -> bool:
    action_lower = action.lower()
    for match in re.finditer(re.escape(action_lower), normalized_lower_text):
        sentence_start = max(
            normalized_lower_text.rfind(".", 0, match.start()),
            normalized_lower_text.rfind("!", 0, match.start()),
            normalized_lower_text.rfind("?", 0, match.start()),
        ) + 1
        sentence_end_candidates = [
            index
            for index in (
                normalized_lower_text.find(".", match.end()),
                normalized_lower_text.find("!", match.end()),
                normalized_lower_text.find("?", match.end()),
            )
            if index != -1
        ]
        sentence_end = (
            min(sentence_end_candidates)
            if sentence_end_candidates
            else len(normalized_lower_text)
        )
        sentence = normalized_lower_text[sentence_start:sentence_end].strip()
        action_offset = match.start() - sentence_start
        clause_start, clause_end = _action_clause_bounds(sentence, action_offset)
        clause = sentence[clause_start:clause_end].strip()
        before_action = sentence[clause_start:action_offset]
        has_negation = (
            re.search(r"\bno\b", before_action) is not None
            or "не " in clause
            or re.search(
                r"\b(?:was|were)\s+not\s+(?:run|sent|executed|performed|applied|used)\b"
                r"|\bcommands?\s+(?:was|were)\s+not\s+(?:run|sent|executed|applied)\b"
                r"|\b(?:did|do|does)\s+not\s+(?:run|send|execute|perform|apply|use)\b"
                r"|\b(?:was|were)\s+not\s+"
                r"(?:accessed|modified|read|opened|inspected|checked|touched)\b"
                r"|\b(?:did|do|does)\s+not\s+"
                r"(?:access|modify|read|open|inspect|check|touch)\b"
                r"|\bnever\s+"
                r"(?:run|sent|executed|performed|applied|used|accessed|modified|read|opened|inspected|checked|touched)\b",
                clause,
            )
            is not None
        )
        has_execution_verb = any(verb in clause for verb in audit_verbs)
        if has_execution_verb and not has_negation:
            return True

    return False


def _action_clause_bounds(sentence: str, action_offset: int) -> tuple[int, int]:
    """Return the contrast/semicolon-delimited clause containing an action."""

    clause_start = 0
    boundary_pattern = re.compile(r";|,\s*(?:but|however|though)\b|\b(?:but|however)\b")
    for boundary in boundary_pattern.finditer(sentence):
        if action_offset < boundary.start():
            return clause_start, boundary.start()
        clause_start = boundary.end()

    return clause_start, len(sentence)


def _remove_heading_section(text: str, heading: str) -> str:
    heading_pattern = re.compile(rf"^{re.escape(heading)}\s*$", re.MULTILINE)
    heading_match = heading_pattern.search(text)
    if not heading_match:
        return text

    heading_level = len(heading) - len(heading.lstrip("#"))
    remaining_text = text[heading_match.end():]
    next_heading = re.search(rf"^#{{1,{heading_level}}}\s+", remaining_text, re.MULTILINE)
    section_end = (
        heading_match.end() + next_heading.start()
        if next_heading
        else len(text)
    )
    return text[:heading_match.start()] + "\n" + text[section_end:]


def validate_report(
    report_path: Path,
    project_root: Path = PROJECT_ROOT,
    *,
    now: datetime | None = None,
) -> list[str]:
    """Return validation errors for one Codex cycle report."""

    errors: list[str] = []
    root = project_root.resolve()
    expected_dir = (root / ".paperclip" / "reports").resolve()
    resolved_report = report_path.resolve()

    if not _is_relative_to(resolved_report, expected_dir):
        errors.append(f"report must be inside {expected_dir}")

    if not REPORT_NAME_PATTERN.match(resolved_report.name):
        errors.append("report filename must match automixer_codex_cycle_YYYYMMDDTHHMMSSZ.md")

    if not resolved_report.is_file():
        errors.append(f"report file does not exist: {resolved_report}")
        return errors

    text = resolved_report.read_text(encoding="utf-8")
    for phrase in REQUIRED_PHRASES:
        if not _contains_required_phrase(text, phrase):
            errors.append(f"missing required report phrase: {phrase}")

    errors.extend(_validate_report_filename_timestamp(resolved_report, now=now))
    errors.extend(_validate_report_title(text, resolved_report))
    errors.extend(_validate_plan_section(text))
    errors.extend(_validate_plan_item_limit(text))
    errors.extend(_validate_required_sections_have_content(text))
    errors.extend(_validate_scope_evidence(text))
    errors.extend(_validate_scope_repo_local_report_path(text, root))
    errors.extend(_validate_scope_project_boundary_contradictions(text))
    errors.extend(_validate_project_boundary_contradictions(text))
    errors.extend(_validate_secret_boundary_contradictions(text))
    errors.extend(_validate_discovery_evidence(text))
    errors.extend(_validate_git_status_and_branch_evidence(text))
    errors.extend(_validate_recent_discovery_evidence(text))
    errors.extend(_validate_todo_scan_boundary_evidence(text))
    errors.extend(_validate_project_goal_evidence(text))
    errors.extend(_validate_tests_and_checks_evidence(text))
    errors.extend(_validate_memory_lint_evidence(text))
    errors.extend(_validate_memory_lint_result_section(text))
    errors.extend(_validate_readiness_scores(text))
    errors.extend(_validate_readiness_scores_are_final_content(text))
    errors.extend(_validate_ordered_core_sections(text))
    errors.extend(_validate_ordered_mid_sections(text))
    errors.extend(_validate_ordered_end_sections(text))
    errors.extend(_validate_mandatory_questions_directly_precede_plain_explanation(text))
    errors.extend(_validate_changed_file_links(text, resolved_report, root))
    errors.extend(_validate_next_steps_limit(text))
    errors.extend(_validate_plain_explanation_is_final_major_section(text))
    errors.extend(_validate_readiness_is_final_plain_subsection(text))
    errors.extend(_validate_plain_explanation_subsections_have_content(text))
    errors.extend(_validate_project_state_explanation_evidence(text))
    errors.extend(_validate_mandatory_question_answers(text))
    errors.extend(_validate_live_safety_evidence(text))

    return errors


def main(argv: Iterable[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a repo-local Automixer Codex cycle report."
    )
    parser.add_argument("report", type=Path, help="Path to automixer_codex_cycle_*.md")
    args = parser.parse_args(list(argv) if argv is not None else None)

    errors = validate_report(
        args.report,
        project_root=project_root,
        now=datetime.now(timezone.utc),
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
