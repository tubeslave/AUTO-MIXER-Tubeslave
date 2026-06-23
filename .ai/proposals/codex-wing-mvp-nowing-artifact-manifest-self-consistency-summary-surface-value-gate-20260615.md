# Proposal: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface Value Gate

## 1. Short Thesis

Convert the manifest self-consistency summary surface from presence-only to
value-checked, fail-closed evidence.

## 2. What I Understood

Offline no-WING handoff artifacts may improve operator readiness, but they must
not create readiness authority. A stale summary line such as live authority
`true` must be rejected before the handoff is treated as clean.

## 3. Proposed Solution

Add a dedicated self-consistency summary-surface check that compares
`summary.json` and `summary.md` against the expected manifest self-consistency
contract. The check writes JSON, Markdown, and combined mismatch lists and fails
the summary when any value drifts.

## 4. Likely Files to Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## 5. Alternatives Considered

Keep checking only line presence. This is cheaper but misses unsafe edited
values. Reuse the manifest sidecar self-consistency check alone. That does not
protect the operator-facing summary surface.

## 6. Risks

The generated no-WING summary grows by several diagnostic fields. Existing
consumers should ignore unknown fields, and the change is limited to no-WING
artifact validation.

## 7. Test Plan

- Targeted summary value-gate regression tests.
- no-WING quick and direct artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Where the Other Agent May Disagree

Kimi may argue that the sidecar is already authoritative. The counterpoint is
that the summary is the operator handoff entrypoint, so stale summary values need
their own fail-closed signal.
