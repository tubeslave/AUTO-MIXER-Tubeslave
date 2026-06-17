# ADR: WING MVP Read-Only Runbook no-WING Manifest Sidecar Checklist

Date: 2026-06-15

## 1. Context

The supervised WING MVP remains blocked on a real read-only WING connection
proof. no-WING release artifacts now expose manifest self-consistency
summary-surface verdicts in `summary.json`, `summary.md`, and the detached
`artifact-manifest.json`, but the read-only runbook did not yet require
operators to inspect that sidecar before opening WING.

## 2. Options Considered

1. Leave the sidecar review implicit in no-WING release reports.
2. Add another artifact field and keep the runbook unchanged.
3. Add an explicit runbook checklist and rails test for the sidecar verdict.

## 3. Decision

Add the runbook checklist and rails test.

## 4. Why This Won

The real hardware procedure should name the exact detached sidecar markers that
must be clean before WING is opened. This keeps the improvement procedural and
non-live while using evidence already produced by no-WING validation.

## 5. Rejected Alternatives

Implicit review was rejected because release reports are not an operator
procedure. A new artifact field was rejected because the sidecar already carries
the verdict.

## 6. Implementation Plan

- Add precondition and fail-closed wording to
  `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Add a rails test for the sidecar checklist markers.
- Register the report in `Docs/INDEX.md` and blocker evidence.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## 7. Test Plan

- Targeted rails test.
- Focused WING/no-WING/Director tests.
- Project Director guard snapshot.
- Memory lint and `git diff --check`.
- Full standard pytest.

## 8. Risks and Rollback

Risk is limited to procedural friction. Roll back by removing the checklist,
test, and evidence report if the manifest sidecar contract changes.
