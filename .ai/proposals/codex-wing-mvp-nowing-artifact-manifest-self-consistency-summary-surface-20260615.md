# Proposal: WING MVP no-WING Artifact Manifest Self-Consistency Summary Surface

## 1. Short Thesis

Surface the manifest self-consistency verdict in the no-WING summary artifacts so
operators and automation can see the sidecar verdict without opening the sidecar
first.

## 2. What I Understood

The first open blocker is still `WING read-only connection proof`. Offline
handoff artifacts can harden the operator path, but they must not close
readiness or imply live authority.

## 3. Proposed Solution

Add the existing manifest self-consistency field set to the required summary
JSON/Markdown surface and render the expected pass/deny values in `summary.md`.
Keep the detached `artifact-manifest.json` as the actual bundle sidecar.

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

Leave the verdict only in `artifact-manifest.json`. This keeps the sidecar
compact but hides the verdict from the operator-readable summary. Add a separate
report artifact. This duplicates state already produced by the release check.

## 6. Risks

The required summary field list gets longer. Existing consumers should ignore
unknown fields, and the change is limited to no-WING release artifacts.

## 7. Test Plan

- Targeted manifest summary surface tests.
- no-WING quick with direct inspection of `summary.json`, `summary.md`, and
  `artifact-manifest.json`.
- Focused WING/no-WING/Director suite.
- Full standard pytest.

## 8. Where the Other Agent May Disagree

Kimi may prefer the manifest sidecar to remain the only source of this verdict.
The counterpoint is operator ergonomics: the summary artifacts are the handoff
entrypoint before a real WING session.
