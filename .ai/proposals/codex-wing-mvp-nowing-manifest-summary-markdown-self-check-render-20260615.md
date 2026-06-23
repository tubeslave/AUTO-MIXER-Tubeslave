# Proposal: no-WING Manifest Summary Markdown Self-Check Render

## Short thesis

Keep the operator-readable no-WING `summary.md` synchronized with the computed
manifest summary surface self-check.

## What I understood

The WING supervised MVP is still waiting for real read-only proof. Offline
no-WING artifacts prepare the operator handoff and must fail closed when their
own summary surfaces are incomplete.

## Proposed solution

- Add a small Markdown line replacement helper for already-rendered summary
  fields.
- After the manifest Markdown surface check runs, replace the
  `artifact_manifest_summary_surface_missing_field_lines` and
  `artifact_manifest_summary_surface_field_lines_rendered` lines with computed
  values.
- Add a regression test that injects a fake required Markdown field line and
  verifies both JSON and `summary.md` show the fail-closed state.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Leave the static success strings in `summary.md`: rejected because an
  operator could see `[]` and `true` even after the check computed missing
  lines.
- Rebuild the whole Markdown writer around templates: rejected as broader than
  needed for this focused gate.

## Risks

The helper updates only fields already present in the summary. That is suitable
for this check because absence is handled by the existing missing-line
validator.

## Test plan

- Targeted manifest summary tests.
- Focused no-WING, Project Director, and WING MVP rails tests.
- no-WING quick check.
- Standard pytest.

## Where the other agent may disagree

Kimi may prefer a declarative summary field renderer. That would reduce future
drift, but the targeted helper closes the current stale Markdown self-report
risk with less churn.
