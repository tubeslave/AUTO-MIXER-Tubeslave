# Proposal: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface Sidecar Binding

## 1. Short Thesis

Record the summary-surface value-gate verdict in the detached manifest sidecar so
the bundle manifest proves the operator handoff summary was value-checked.

## 2. What I Understood

The first open blocker is still `WING read-only connection proof`. Offline
artifacts can improve the handoff path, but cannot close readiness or grant live
authority.

## 3. Proposed Solution

When writing `artifact-manifest.json`, read the generated `summary.json` and
`summary.md`, recompute the manifest self-consistency summary-surface value
gate, and include the diagnostic fields in the sidecar.

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

Leave the verdict only in summaries. This keeps the sidecar smaller but weakens
the manifest as a detached bundle index. Add a new artifact. That duplicates the
same state.

## 6. Risks

The manifest schema gains more diagnostic fields. Consumers should ignore
unknown fields, and the change is isolated to no-WING artifacts.

## 7. Test Plan

- Targeted no-WING manifest tests.
- no-WING quick and direct artifact inspection.
- Focused WING/no-WING/Director tests.
- Full standard pytest.

## 8. Where the Other Agent May Disagree

Kimi may prefer summary-only diagnostics. The counterpoint is that the detached
manifest already binds summary hashes, so it is the correct place to mirror the
summary value-gate verdict.
