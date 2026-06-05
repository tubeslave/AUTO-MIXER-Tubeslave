# cycle-report-plain-explanation-adjacency-20260605

## Context

Codex autoloop reports are part of the project memory surface. The required
"three questions" section should flow directly into the plain-language
explanation so human operators can find the summary without a new top-level
section splitting the end of the report.

## Goal

Harden `tools/codex_autoloop/validate_cycle_report.py` so it rejects reports
that insert another `##` section between `## Три обязательных вопроса` and
`## Объяснение простым языком`.

## Non-goals

- Do not change live mixer behavior.
- Do not change report generation scheduling.
- Do not rewrite existing cycle reports.

## Safety constraints

- This is reporting/tooling only.
- The validator must not imply live OSC/WING readiness.
- Existing order checks should remain in force.

## Expected evidence

- A focused pytest covers the new rejection.
- Full backend pytest remains green.
- Memory lint remains green.
