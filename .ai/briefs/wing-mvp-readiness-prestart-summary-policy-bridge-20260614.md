# WING MVP Readiness Prestart Summary Policy Bridge

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
Accepted read-only candidates already carry a no-WING prestart summary type and
connection policy. The autonomy promotion readiness bridge surfaces prestart
path, SHA, status, and no-contact flags for manual review, but it must also
surface the type and no-hardware/no-network/no-live-write policy contract.

## Task

Expose the accepted-candidate prestart summary type and connection-policy
contract matches in `backend/autonomy_promotion_readiness.py` and cover them in
focused readiness tests.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline or candidate evidence.
- Keep the selected blocker as `WING read-only connection proof`.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused autonomy promotion readiness tests.
- Run focused WING/readiness rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director JSON output and direction guard.
- Run `py_compile`, memory lint, `git diff --check`, and the standard full
  pytest command before commit.
