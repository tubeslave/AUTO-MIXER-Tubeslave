# ADR: WING MVP Backend Prestart Summary JSON Surface

Date: 2026-06-15

## 1. Context

The no-WING handoff packet artifact already binds the intended backend/WING
session to a passed no-WING prestart `summary.json` by path, SHA256, type,
status, connection policy, and safety flags. Top-level `summary.json` did not
expose the same compact machine-readable binding.

## 2. Options Considered

1. Keep the values nested only.
2. Add a separate backend-prestart binding artifact.
3. Project the binding into top-level no-WING `summary.json`.

## 3. Decision

Project the backend prestart summary binding into top-level `summary.json`.

## 4. Why This Won

The no-WING summary is the primary release-gate surface. Consumers should be
able to verify the bound prestart summary path, SHA, status, policy, and safety
verdicts without parsing a large nested packet artifact.

## 5. Rejected Alternatives

Nested-only data was rejected because it hides a required pre-backend gate. A
separate artifact was rejected because it adds another file contract without
changing the source of truth.

## 6. Implementation Plan

- Build a top-level backend prestart summary projection from the handoff
  artifact.
- Add assertions for projected JSON fields and denied authority.
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
projection, test assertions, and evidence docs if the prestart binding contract
is replaced.
