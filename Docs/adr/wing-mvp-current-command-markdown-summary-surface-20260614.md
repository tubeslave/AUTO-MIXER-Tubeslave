# ADR: WING MVP Current Command Markdown Summary Surface

## 1. Context

The no-WING release check renders required and missing supervised proof-plan
summary fields in the operator-readable `summary.md`. Project Director baseline
coverage already checked for the underlying proof-plan summary fields, but did
not explicitly require the markdown surface that an operator reviews.

## 2. Options considered

- Keep the existing machine-readable field checks only.
- Require a separate new schema field for markdown rendering.
- Add an explicit Current First Command baseline marker for the operator-readable
  no-WING `summary.md` proof-plan section.

## 3. Decision

Add the explicit Current First Command baseline marker and mirror it in the
autoloop prompt, README, and tests.

## 4. Why this won

The supervised WING MVP handoff must be inspectable by a human before physical
proof work. A small baseline marker protects that operator surface without
adding runtime dependencies or touching live-control code.

## 5. Rejected alternatives

Machine-only checks were rejected because they do not prove the operator-facing
handoff is readable. A new schema field was rejected because the data already
exists and the missing contract is presentation.

## 6. Implementation plan

- Add `operator-readable no-WING summary.md` to baseline marker coverage.
- Update Current First Command and autoloop docs with the same requirement.
- Add focused tests for Project Director selected action, prompt, README, and
  missing-marker behavior.

## 7. Test plan

Run the focused WING rail and Project Director tests, plus diff whitespace and
memory lint. No hardware or network access is required.

## 8. Risks and rollback

Risk is limited to stale autoloop wording or marker drift. Roll back by removing
the marker and matching docs/tests if the wording proves too narrow. This ADR
does not grant live WING authority.
