# Proposal: WING MVP no-WING artifact manifest sidecar

## 1. Short thesis

Generate a detached no-WING artifact manifest after `summary.json` and
`summary.md` are written.

## 2. What I understood

The WING handoff artifacts should be easy for an operator or reviewer to verify.
`summary.json` cannot safely contain its own SHA, so the bundle needs a sidecar
manifest for whole-file integrity.

## 3. Proposed solution

Add `artifact-manifest.json` under each no-WING run directory. It records path,
byte count, and SHA256 for `summary.json` and `summary.md`, plus explicit
no-readiness/no-live-authority flags.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- WING rail tests and docs

## 5. Alternatives considered

- Put `summary.json` SHA inside `summary.json`.
- Rely only on the markdown digest already inside `summary.json`.

## 6. Risks

The sidecar must not be mistaken for readiness closure. The manifest therefore
declares `artifact_manifest_readiness_closure_allowed: false` and
`artifact_manifest_live_authority_granted: false`.

## 7. Test plan

Run focused no-WING summary tests, Project Director/WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer one global manifest for the whole handoff packet. A per-run
manifest is smaller and directly protects the artifacts generated here.
