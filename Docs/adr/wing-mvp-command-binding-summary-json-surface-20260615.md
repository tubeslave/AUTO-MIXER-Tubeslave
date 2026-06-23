# ADR: WING MVP Command Binding Summary JSON Surface

Date: 2026-06-15

## 1. Context

The no-WING handoff packet artifact already verifies preview, capture, review,
and decision-review command bindings. Operator Markdown renders those verdicts.
Top-level `summary.json` did not expose the same compact machine-readable
command binding surface.

## 2. Options Considered

1. Keep the values nested only.
2. Add a separate command-binding artifact.
3. Project the command binding verdicts into top-level no-WING `summary.json`.

## 3. Decision

Project the WING read-only command binding verdicts into top-level
`summary.json`.

## 4. Why This Won

The no-WING summary is the primary release-gate machine surface. Consumers
should be able to verify preview/capture/review/decision bindings without
parsing the nested handoff packet.

## 5. Rejected Alternatives

Nested-only data was rejected because it hides pre-live gates from simple
automation. A separate artifact was rejected because the values are already
validated inside the summary pipeline.

## 6. Implementation Plan

- Build a top-level command binding projection from the handoff artifact.
- Add assertions for the projected JSON fields and denied authority.
- Register the evidence and regenerate latest MVP progress.

## 7. Test Plan

- Targeted no-WING summary test.
- no-WING quick.
- Project Director guard snapshot.
- Focused WING/no-WING/Director tests.
- Full standard pytest.
- Memory lint and `git diff --check`.

## 8. Risks and Rollback

Risk is limited to duplicated reporting fields. Roll back by removing the
projection, test assertions, and evidence docs if the command binding contract
is replaced.
