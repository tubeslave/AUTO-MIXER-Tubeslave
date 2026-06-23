# ADR: WING MVP Backend Prestart Gate Summary JSON Surface

Date: 2026-06-15

## 1. Context

The no-WING handoff packet artifact already exposes backend-port prestart gate
match verdicts and the prestart check command text. The operator-readable
Markdown summary renders those verdicts. Top-level `summary.json` did not expose
the same compact machine-readable gate surface.

## 2. Options Considered

1. Keep the values nested only.
2. Add a separate backend-prestart summary artifact.
3. Project the values into top-level no-WING `summary.json`.

## 3. Decision

Project the backend prestart gate verdicts and command text into top-level
`summary.json`.

## 4. Why This Won

The no-WING summary is the release gate's primary machine-readable surface.
Gate consumers should be able to verify backend prestart gate match, declared
match, and command text without parsing the nested handoff artifact.

## 5. Rejected Alternatives

Nested-only data was rejected because it hides a required pre-backend gate
behind a large packet field. A separate artifact was rejected because the values
are already part of the summary contract and do not need another file.

## 6. Implementation Plan

- Build a top-level backend prestart gate summary projection from the handoff
  artifact.
- Add assertions for the projected JSON fields.
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
projection, test assertions, and evidence docs if the handoff gate contract is
replaced.
