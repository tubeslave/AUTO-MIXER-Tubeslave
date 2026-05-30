#!/usr/bin/env python3
"""Validate the required shape of a repo-local Codex cycle report."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME_PATTERN = re.compile(r"^automixer_codex_cycle_\d{8}T\d{6}Z\.md$")
REPORT_TITLE_PATTERN = re.compile(
    r"^#\s+Automixer Codex Cycle\s+(\d{8}T\d{6}Z)\s*$",
    re.MULTILINE,
)
MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]\n]+\]\(([^)\n]+)\)")

REQUIRED_PHRASES = (
    "## What Was Found",
    "## What Was Done",
    "## Tests And Checks",
    "## What Remains",
    "## Risks",
    "## Changed Files",
    "Live safety status:",
    "no live OSC/WING writes were executed",
    "External Paperclip/PiperClip was not used as an agent runner",
    "No neighboring repositories or external local projects were analyzed, modified, or used.",
    "Repo-local `.paperclip/reports/` was used only as an Automixer report directory.",
    "## Три обязательных вопроса",
    "1. Что нового проект научился делать?",
    "2. Что сейчас является самым слабым местом?",
    "3. Что даст наибольший прогресс в следующем цикле?",
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
    "## What Was Found",
    "## What Was Done",
    "## Tests And Checks",
    "## What Remains",
    "## Risks",
)

PLAN_SECTION_HEADINGS = (
    "## Short Plan",
    "## Plan Selected",
)
MAX_PLAN_ITEMS = 3

MANDATORY_QUESTIONS_HEADING = "## Три обязательных вопроса"
MANDATORY_QUESTIONS = (
    "1. Что нового проект научился делать?",
    "2. Что сейчас является самым слабым местом?",
    "3. Что даст наибольший прогресс в следующем цикле?",
)

REQUIRED_LIVE_SAFETY_PHRASES = (
    "No real console was contacted",
)

REQUIRED_NON_DESTRUCTIVE_PHRASES = (
    "No commit",
    "push",
    "reset",
    "destructive cleanup",
    "secret access",
    "user-file deletion",
)

FORBIDDEN_LIVE_COMMANDS = (
    "set_fader",
    "set_gain",
    "set_eq",
    "set_compressor",
    "route_output",
    "load_snap",
    "save_snap",
    "reset",
    "mute",
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_readiness_scores(text: str) -> list[str]:
    errors: list[str] = []

    for label in REQUIRED_READINESS_SCORES:
        pattern = re.compile(
            rf"^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(\d{{1,3}})\s*%\s*$",
            re.MULTILINE,
        )
        match = pattern.search(text)
        if not match:
            errors.append(f"missing readiness score: {label}: N%")
            continue

        value = int(match.group(1))
        if value > 100:
            errors.append(f"readiness score must be between 0 and 100: {label}: {value}%")

    return errors


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
        section_body = _extract_heading_section(text, heading)
        if not section_body:
            continue

        plan_items = [
            line
            for line in section_body.splitlines()
            if re.match(r"^\s*(?:[-*+]\s+|\d+[\.)]\s+)", line)
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


def _validate_next_steps_limit(text: str) -> list[str]:
    section_body = _extract_heading_section(text, NEXT_STEPS_HEADING)
    if not section_body:
        return []

    step_items = [
        line
        for line in section_body.splitlines()
        if re.match(r"^\s*(?:[-*+]\s+|\d+[\.)]\s+)", line)
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
    lower_text = text.lower()
    normalized_lower_text = _collapse_whitespace(text).lower()

    for phrase in REQUIRED_LIVE_SAFETY_PHRASES:
        if phrase.lower() not in normalized_lower_text:
            errors.append(f"missing required live-safety evidence: {phrase}")

    missing_commands = [
        command
        for command in FORBIDDEN_LIVE_COMMANDS
        if command.lower() not in lower_text
    ]
    if missing_commands:
        errors.append(
            "missing forbidden command audit evidence: "
            + ", ".join(missing_commands)
        )

    missing_non_destructive_phrases = [
        phrase
        for phrase in REQUIRED_NON_DESTRUCTIVE_PHRASES
        if phrase.lower() not in normalized_lower_text
    ]
    if missing_non_destructive_phrases:
        errors.append(
            "missing non-destructive action audit evidence: "
            + ", ".join(missing_non_destructive_phrases)
        )

    return errors


def validate_report(report_path: Path, project_root: Path = PROJECT_ROOT) -> list[str]:
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

    errors.extend(_validate_report_title(text, resolved_report))
    errors.extend(_validate_plan_section(text))
    errors.extend(_validate_plan_item_limit(text))
    errors.extend(_validate_required_sections_have_content(text))
    errors.extend(_validate_readiness_scores(text))
    errors.extend(_validate_ordered_end_sections(text))
    errors.extend(_validate_changed_file_links(text, resolved_report, root))
    errors.extend(_validate_next_steps_limit(text))
    errors.extend(_validate_plain_explanation_is_final_major_section(text))
    errors.extend(_validate_mandatory_question_answers(text))
    errors.extend(_validate_live_safety_evidence(text))

    return errors


def main(argv: Iterable[str] | None = None, project_root: Path = PROJECT_ROOT) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a repo-local Automixer Codex cycle report."
    )
    parser.add_argument("report", type=Path, help="Path to automixer_codex_cycle_*.md")
    args = parser.parse_args(list(argv) if argv is not None else None)

    errors = validate_report(args.report, project_root=project_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
