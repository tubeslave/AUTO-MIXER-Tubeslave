# Proposal: WING MVP no-WING artifact manifest summary surface self-check

## 1. Short thesis

Add a fail-closed self-check for the no-WING artifact manifest summary surface.

## 2. What I understood

The manifest contract is visible in `summary.json` and `summary.md`, but
automation should detect if the markdown surface stops rendering required
operator-facing fields.

## 3. Proposed solution

Add required markdown field-line constants and a summary helper that records
`artifact_manifest_summary_surface_missing_field_lines` plus
`artifact_manifest_summary_surface_field_lines_rendered`. If any required line
is absent, append a named failed check and mark the summary failed.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- WING rail tests and docs

## 5. Alternatives considered

- Rely on existing JSON assertions only.
- Add a separate sidecar schema validator.

## 6. Risks

The main risk is adding another marker family to an already long Current First
Command. The benefit is a concrete fail-closed operator-surface check.

## 7. Test plan

Run focused no-WING summary tests, Project Director/WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a normalized schema validator. The inline self-check is smaller
and matches the existing operator markdown proof-plan check.
