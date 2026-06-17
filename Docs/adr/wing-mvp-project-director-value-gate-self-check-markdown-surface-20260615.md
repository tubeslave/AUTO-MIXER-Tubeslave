# ADR: WING MVP Project Director Value Gate Self-Check Markdown Surface

## 1. Context

Project Director selected-action value-gate self-check fields are present in
the no-WING JSON summary. The operator-facing `summary.md` must carry the same
critical fields before a real WING read-only preflight; otherwise a human
reviewer can miss stale selected-action coverage.

## 2. Options considered

- Rely on JSON summary fields.
- Rely on tests that inspect the Markdown renderer.
- Add a runtime no-WING Markdown-surface check.

## 3. Decision

Add a runtime no-WING Markdown-surface check for the Project Director selected
action value-gate self-check fields.

## 4. Why this won

The real WING preflight gate is an operator workflow. Runtime validation should
fail before hardware work if the operator-readable Markdown omits required
Project Director safety fields.

## 5. Rejected alternatives

JSON-only validation was rejected because the runbook references Markdown
handoff review. Test-only coverage was rejected because runtime artifacts should
self-report surface drift.

## 6. Implementation plan

- Add a required field-line tuple and dedicated check name.
- Add an applicator that updates summary status and failed checks on missing
  Markdown lines.
- Render the Markdown-surface check result.
- Extend tests and Project Director baseline markers.

## 7. Test plan

- Unit failing test for a missing Markdown line.
- Integration summary test for JSON and Markdown fields.
- Project Director selected-action marker assertions.
- no-WING quick, focused WING/no-WING suite, full pytest, memory lint, and diff
  check.

## 8. Risks and rollback

Risk: stricter no-WING validation may fail on Markdown drift. Rollback is to
remove the applicator and baseline markers while retaining the JSON-level
Project Director self-check.
