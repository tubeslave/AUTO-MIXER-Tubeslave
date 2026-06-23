# Proposal: WING MVP no-WING artifact manifest summary surface

## 1. Short thesis

Surface the artifact manifest contract in the main no-WING summaries while
leaving the detached manifest as the final hash authority.

## 2. What I understood

The sidecar contains the manifest details, but the operator usually starts from
`summary.json` or `summary.md`. Those summaries should declare which artifact
manifest fields are required without trying to hash the sidecar itself.

## 3. Proposed solution

Add shared manifest-contract constants and render the required artifact count,
IDs, paths, SHA field names, and missing-field status into `summary.json` and
`summary.md`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- WING rail tests and docs

## 5. Alternatives considered

- Require operators to open `artifact-manifest.json` only.
- Add a hash of `artifact-manifest.json` to `summary.json`.

## 6. Risks

The main risk is implying recursive hash closure. The implementation avoids that
by exposing required manifest fields in the summary while keeping final file
hashes in the detached sidecar.

## 7. Test plan

Run focused no-WING summary tests, Project Director/WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a separate schema file for the sidecar. The summary surface is
smaller and directly helps operator review.
