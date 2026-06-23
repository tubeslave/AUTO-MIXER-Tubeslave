# Proposal: WING MVP no-WING operator markdown summary SHA binding

## 1. Short thesis

Record a JSON-only SHA256 binding from no-WING `summary.json` to the exact
operator-readable `summary.md` artifact.

## 2. What I understood

The no-WING handoff bundle is an offline operator gate. It should be easy to
verify that machine-readable status fields refer to the markdown surface the
operator actually saw.

## 3. Proposed solution

Add a small helper in `tools/no_wing_release_check.py` that runs after final
markdown rendering. It writes `operator_markdown_summary_artifact`,
`operator_markdown_summary_sha256`, and
`operator_markdown_summary_sha256_available` under `next_wing_proof_plan`.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- WING rail tests and docs

## 5. Alternatives considered

- Render the SHA inside `summary.md`.
- Add a separate validator command.

## 6. Risks

Rendering the SHA into markdown would create a recursive digest problem. A
separate validator adds process weight without improving the handoff bundle.

## 7. Test plan

Run focused no-WING summary tests, WING rail tests, full pytest, `git diff
--check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may prefer a detached manifest file for artifact digests. That is useful
later, but JSON-only binding is the smallest safe step for the current blocker.
