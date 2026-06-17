# Proposal: WING MVP Read-Only Runbook no-WING Manifest Sidecar Checklist

## 1. Short Thesis

Require the operator-facing WING read-only preflight runbook to verify the
detached no-WING `artifact-manifest.json` sidecar before any backend/WING
WebSocket session opens.

## 2. What I Understood

The no-WING release artifacts now mirror manifest self-consistency value-gate
verdicts across `summary.json`, `summary.md`, and `artifact-manifest.json`.
The runbook still only told operators to keep the passed `summary.json` path,
so a real WING preflight could skip the detached sidecar review.

## 3. Proposed Solution

Add a precondition and fail-closed rule to
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` requiring the no-WING
`artifact-manifest.json` to show the same passing summary-surface verdict and
explicit readiness/live-authority denial fields before opening WING. Add a rails
test that asserts those markers remain present.

## 4. Likely Files to Touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## 5. Alternatives Considered

Leave the checklist implicit in no-WING reports. That weakens the real hardware
procedure because operators should not infer a required sidecar review from
release notes. Add another no-WING artifact field. That would duplicate the
existing sidecar verdict instead of making the runbook consume it.

## 6. Risks

The runbook grows another checklist item. The benefit is that a real WING
preflight cannot bypass detached manifest drift evidence.

## 7. Test Plan

- Targeted rails test for the new runbook sidecar checklist.
- Focused WING/no-WING/Director tests.
- Project Director guard snapshot.
- Memory lint and `git diff --check`.
- Full standard pytest.

## 8. Where the Other Agent May Disagree

Kimi may prefer waiting until a real WING session exists. The counterpoint is
that runbook gates should be hardened before the physical session so operators
know exactly what to review.
