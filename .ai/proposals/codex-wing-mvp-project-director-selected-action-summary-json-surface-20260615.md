# Proposal: WING MVP Project Director Selected-Action Summary JSON Surface

## 1. Short Thesis

Project Director selected-action evidence should be visible as top-level
machine-readable no-WING `summary.json` fields, including whether the current
command contains the manifest sidecar checklist.

## 2. What I Understood

The current first command now carries the manifest sidecar checklist, and
`summary.md` renders the nested Project Director anchor. Machine consumers still
need to walk nested handoff artifact structures to verify the selected action.

## 3. Proposed Solution

Add a no-WING summary projection that copies selected-action source, SHA,
current-command match flags, baseline coverage, and sidecar checklist missing
markers from the handoff artifact into top-level `summary.json`.

## 4. Likely Files to Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `.ai/briefs/`
- `.ai/proposals/`
- `Docs/adr/`
- `Docs/reports/`

## 5. Alternatives Considered

Leave the data nested. That keeps JSON smaller but makes automated checks more
fragile. Add another artifact file. That duplicates a compact projection that
belongs in the release summary.

## 6. Risks

The summary gains more top-level fields. Existing consumers should ignore
unknown fields, and the values are derived from already generated no-WING
artifact data.

## 7. Test Plan

- Targeted no-WING summary test.
- no-WING quick artifact inspection.
- Focused WING/no-WING/Director tests.
- Project Director guard snapshot.
- Memory lint and `git diff --check`.
- Full standard pytest.

## 8. Where the Other Agent May Disagree

Kimi may prefer keeping nested data only. The counterpoint is that top-level
summary fields are the stable machine surface for release gates.
