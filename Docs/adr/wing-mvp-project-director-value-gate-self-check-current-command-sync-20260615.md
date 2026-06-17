# ADR: WING MVP Project Director Value Gate Self-Check Current Command Sync

## 1. Context

No-WING summaries now expose and self-check the accepted-candidate expected
artifact trace value-gate surface. Project Director selects the standalone
`## Current First Command` as its selected action when that command covers the
baseline. That command must include the new self-check requirements, otherwise
the selected action can lag behind the actual handoff contract.

## 2. Options considered

- Keep the markers only in the blocker table next-action cell.
- Add a prose-only reminder to the progress report.
- Require the value-gate self-check markers in the current-first baseline and
  selected action.

## 3. Decision

Require the value-gate self-check fields in `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`
and add them to `## Current First Command`.

## 4. Why this won

Project Director is the operator-facing audit/planning directive. Its selected
action must be synchronized with the no-WING validation contract so the next
backend/WING session cannot be guided by stale instructions.

## 5. Rejected alternatives

Keeping the markers only in the table row was rejected because Project Director
selects the standalone command. Prose-only reporting was rejected because the
baseline coverage check needs machine-checkable markers.

## 6. Implementation plan

- Extend the baseline marker tuple with the six value-gate self-check fields.
- Add those markers to `## Current First Command`.
- Extend Project Director tests for selected action coverage.
- Regenerate `MVP_PROGRESS_LATEST.md`.

## 7. Test plan

- Targeted Project Director selected-action test.
- no-WING quick artifact check.
- Project Director CLI check.
- Focused WING/no-WING suite.
- Full pytest, memory lint, and whitespace diff check.

## 8. Risks and rollback

Risk: current-first command baseline coverage becomes stricter and may fail if
the command is edited without preserving the self-check fields. Rollback is to
remove the new markers from the tuple and command while keeping the no-WING
summary self-check itself.
