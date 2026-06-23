# Codex Proposal: WING MVP no-WING Operator Markdown Summary Self-Check

## 1. Short thesis

Add a no-WING artifact self-check that proves the operator-readable `summary.md`
proof-plan section rendered the required supervised proof-plan field lines.

## 2. What I understood

The rail already requires the markdown surface. The artifact bundle should also
report whether that markdown surface was actually rendered, and fail closed if
the required field lines disappear.

## 3. Proposed solution

After building `summary.md`, compute a small self-check over the rendered text.
Store the verdict under `next_wing_proof_plan` and require the verdict in the
Project Director current-first-command baseline.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`

## 5. Alternatives considered

- Rely only on markdown rendering tests. Rejected because the generated artifact
  would not carry a self-describing verdict for operators and reviewers.
- Add a separate command. Rejected because the check belongs inside summary
  generation and does not need another execution path.

## 6. Risks

The main risk is marker drift. Keeping the field list short and tested makes the
contract straightforward.

## 7. Test plan

Run focused no-WING summary tests, Project Director rail tests, `git diff
--check`, and memory lint.

## 8. Where the other agent may disagree

Kimi may consider the self-check redundant with direct markdown assertions. The
counterpoint is that the artifact bundle itself should carry the handoff verdict.
