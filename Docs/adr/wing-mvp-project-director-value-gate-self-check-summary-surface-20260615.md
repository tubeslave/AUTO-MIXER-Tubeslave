# ADR: WING MVP Project Director Value Gate Self-Check Summary Surface

## 1. Context

Project Director selected action is now synchronized with the value-gate
summary-surface self-check requirements. However, no-WING handoff summaries
only exposed broader selected-action digest and baseline fields. Operators
needed a direct top-level verdict that the selected action still contains the
value-gate self-check markers.

## 2. Options considered

- Rely on selected-action digest matching alone.
- Keep the check inside Project Director runtime output only.
- Add a dedicated no-WING summary surface for the marker coverage.

## 3. Decision

Add a dedicated Project Director selected-action value-gate self-check summary
surface to no-WING validation.

## 4. Why this won

Digest checks prove identity but do not tell an operator which safety contract
is missing when the selected action drifts. The new surface gives a direct
missing-marker list, a pass flag, and explicit readiness/live-authority denial
fields.

## 5. Rejected alternatives

Digest-only validation was rejected because it is less actionable during
operator review. Runtime-only reporting was rejected because the no-WING handoff
summary is the release gate visible before the WING session.

## 6. Implementation plan

- Add the marker tuple and missing-marker helper.
- Fail handoff validation when selected-action value-gate self-check markers
  are missing.
- Render the marker coverage in `summary.json` and operator `summary.md`.
- Require the same field names in the current-first baseline command.

## 7. Test plan

- Passing no-WING handoff summary assertions for JSON and Markdown fields.
- Failing handoff packet test with one marker removed.
- Project Director selected-action coverage test.
- Full no-WING/Project Director focused validation and full pytest.

## 8. Risks and rollback

Risk: stricter current-first baseline coverage can fail if the command is edited
without the new markers. Rollback is to remove the marker tuple, summary fields,
and current-first baseline additions while keeping the existing digest checks.
