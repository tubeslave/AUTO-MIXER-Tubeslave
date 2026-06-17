# ADR: WING MVP Accepted Candidate Artifact Root Path Scope

## 1. Context

The accepted-candidate artifact-root gate protects the transition from a
reviewed read-only WING preflight candidate to physical supervised proof
planning. The existing helper checked only that the accepted source id was
under `artifact_root`, while related summary and event artifact fields could
drift.

## 2. Options Considered

- Keep duplicate local helper logic in planner and no-WING validation.
- Add a broad candidate-review schema migration.
- Add a shared artifact-root path-scope contract helper.

## 3. Decision

Use a shared contract helper that validates all accepted candidate artifact path
fields against the declared artifact root.

## 4. Why This Won

The shared helper prevents planner and release validation from drifting. It is
small, offline, and preserves all no-live-authority semantics.

## 5. Rejected Alternatives

Duplicate helper logic was already present and weaker. A broad schema migration
would be larger than needed for the current first blocker.

## 6. Implementation Plan

- Add a shared accepted-candidate artifact-root helper to the contract module.
- Use it in supervised proof planning.
- Use it in no-WING next-proof-plan validation.
- Add regressions for artifact paths that leave the declared root while
  prestart remains valid.

## 7. Test Plan

Run focused tests for supervised proof planning, no-WING validation, and review
decision contract helpers. Then run the WING-focused suite, standard pytest,
no-WING release check, diff check, memory lint, and Project Director.

## 8. Risks And Rollback

Risk is limited to rejecting malformed candidate review data before proof
planning. Rollback is reverting the shared helper and returning both call sites
to their previous local checks.
