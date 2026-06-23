# Brief: WING MVP no-WING Manifest Cross-Surface Safety Denial

## Task

Harden the no-WING artifact manifest cross-surface contract so `summary.json`
and operator-readable `summary.md` explicitly prove that manifest
cross-surface readiness closure and live authority remain denied.

## Context

The current first MVP blocker is `WING read-only connection proof`. Offline
no-WING evidence can prepare the operator handoff, but it must not close
readiness or grant live authority. The manifest cross-surface check already
compares contract id and digest fields across surfaces; it should also fail
closed if the safety denial fields drift to `true` in either primary surface.

## Scope

- Add computed cross-surface verdicts for JSON readiness-closure denial.
- Add computed cross-surface verdicts for JSON live-authority denial.
- Add computed cross-surface verdicts for Markdown readiness-closure denial.
- Add computed cross-surface verdicts for Markdown live-authority denial.
- Keep the change offline and no-WING only.

## Non-Goals

- Do not contact WING.
- Do not query or mutate any console state.
- Do not close the WING readiness blocker.
- Do not grant supervised or autonomous live authority.

## Validation

- Targeted no-WING manifest tests.
- Focused WING/no-WING/Director tests.
- Full standard pytest before merge.
